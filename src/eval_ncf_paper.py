"""
Replicates the evaluation protocol from the original Neural Collaborative
Filtering paper (He et al. 2017, "Neural Collaborative Filtering", WWW):

1. Leave-one-out split: for each user, their MOST RECENT interaction
   (by timestamp) is treated as the held-out test item
2. For each user, sample 99 random items they've never interacted with
   as negatives
3. Rank the test item among those 99 negatives + itself (100 candidates
   total) using the model's predicted score
4. HR@10 (Hit Rate) = 1 if the test item lands in the top 10 of that
   ranking, else 0 -- averaged across all evaluated users
5. NDCG@10 (Normalized Discounted Cumulative Gain) = rewards the test
   item landing HIGHER in the top 10, not just anywhere in it -- a hit
   at rank 1 scores higher than a hit at rank 10

IMPORTANT CAVEAT -- read before trusting this number:
Our NCF model (src/collaborative.py) was trained on a RANDOM 90/5/5
split, not a leave-one-out-by-timestamp split. That means for some
users, the "latest interaction" this script treats as held-out test
data may have actually been INSIDE the training set already. This can
inflate HR@10/NDCG@10 above what a fair, leak-free evaluation would show.
This script is useful for understanding the protocol and getting a rough
comparison point, but is NOT a strictly fair apples-to-apples comparison
against the original paper's numbers unless the model is retrained with
a proper leave-one-out split (see retrain_leave_one_out() below for that).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.collaborative import NeuMF

NUM_NEGATIVES = 99   # matches the paper: 1 positive + 99 negatives = 100 candidates
TOP_K = 10
NUM_EVAL_USERS = 1000  # paper evaluates all users; we sample for speed -- bump this up if you have time


def load_model_and_mappings(device):
    checkpoint = torch.load(config.MODELS_DIR / "ncf_model.pt",
                             weights_only=False, map_location=device)
    model = NeuMF(
        checkpoint["num_users"], checkpoint["num_movies"],
        embed_dim=checkpoint["embed_dim"], mlp_layers=checkpoint["mlp_layers"],
        global_mean=checkpoint.get("global_mean", 3.5),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    id_maps = np.load(config.MODELS_DIR / "ncf_id_mappings.npz")
    user_to_idx = {u: i for i, u in enumerate(id_maps["user_ids"])}
    movie_to_idx = {m: i for i, m in enumerate(id_maps["movie_ids"])}
    return model, user_to_idx, movie_to_idx


def build_leave_one_out_split(ratings: pd.DataFrame):
    """
    For each user, find their single most recent rating (by timestamp) --
    that's the held-out test item. Returns a dict: {userId: test_movieId}
    and a dict: {userId: set of ALL movieIds they've ever rated} (used to
    make sure sampled negatives are genuinely items they've never touched).
    """
    ratings_sorted = ratings.sort_values("timestamp")
    test_items = ratings_sorted.groupby("userId").tail(1).set_index("userId")["movieId"].to_dict()
    all_interacted = ratings.groupby("userId")["movieId"].apply(set).to_dict()
    return test_items, all_interacted


def sample_negatives(user_interacted: set, all_movie_ids: np.ndarray,
                      num_negatives: int, rng: np.random.Generator) -> list:
    """Sample movies this user has never rated, as negative candidates."""
    negatives = []
    attempts = 0
    max_attempts = num_negatives * 20  # safety valve against infinite loop
    while len(negatives) < num_negatives and attempts < max_attempts:
        candidate = all_movie_ids[rng.integers(0, len(all_movie_ids))]
        if candidate not in user_interacted:
            negatives.append(candidate)
        attempts += 1
    return negatives


def evaluate_hr_ndcg(model, user_to_idx, movie_to_idx, test_items, all_interacted,
                      eval_users, device, top_k=TOP_K, num_negatives=NUM_NEGATIVES):
    all_movie_ids = np.array(list(movie_to_idx.keys()))
    rng = np.random.default_rng(42)

    hits, ndcgs, evaluated = 0, 0.0, 0

    for user_id in tqdm(eval_users, desc="Evaluating"):
        if user_id not in user_to_idx or user_id not in test_items:
            continue
        test_movie_id = test_items[user_id]
        if test_movie_id not in movie_to_idx:
            continue  # test item got filtered out during preprocessing, skip

        negatives = sample_negatives(all_interacted[user_id], all_movie_ids,
                                      num_negatives, rng)
        if len(negatives) < num_negatives:
            continue  # couldn't find enough negatives (rare edge case), skip

        candidates = [test_movie_id] + negatives
        candidate_indices = [movie_to_idx[m] for m in candidates]
        user_idx = user_to_idx[user_id]

        with torch.no_grad():
            u_tensor = torch.full((len(candidates),), user_idx, dtype=torch.long, device=device)
            m_tensor = torch.tensor(candidate_indices, dtype=torch.long, device=device)
            scores = model(u_tensor, m_tensor).cpu().numpy()

        # rank candidates by predicted score, descending; test item is index 0
        ranking = np.argsort(-scores)
        rank_of_test_item = int(np.where(ranking == 0)[0][0]) + 1  # 1-indexed rank

        evaluated += 1
        if rank_of_test_item <= top_k:
            hits += 1
            ndcgs += 1.0 / np.log2(rank_of_test_item + 1)  # NDCG: rank 1 scores higher than rank 10

    hr = hits / evaluated if evaluated else 0.0
    ndcg = ndcgs / evaluated if evaluated else 0.0
    return hr, ndcg, evaluated


def run_paper_protocol_eval():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("\n*** CAVEAT: our model used a random 90/5/5 split, not a proper")
    print("*** leave-one-out-by-timestamp split. This number may be inflated")
    print("*** vs. a fully fair comparison -- read the module docstring. ***\n")

    model, user_to_idx, movie_to_idx = load_model_and_mappings(device)
    ratings = pd.read_parquet(config.DATA_PROCESSED / "ratings_clean.parquet")

    print("Building leave-one-out split (this scans all ratings once)...")
    test_items, all_interacted = build_leave_one_out_split(ratings)

    rng = np.random.default_rng(42)
    known_users = [u for u in test_items.keys() if u in user_to_idx]
    n = min(NUM_EVAL_USERS, len(known_users))
    eval_users = rng.choice(known_users, size=n, replace=False).tolist()
    print(f"Evaluating on {n} users (sampled from {len(known_users)} eligible)\n")

    hr, ndcg, evaluated = evaluate_hr_ndcg(
        model, user_to_idx, movie_to_idx, test_items, all_interacted, eval_users, device
    )

    print(f"\n=== Results (paper-style protocol, 1 positive + {NUM_NEGATIVES} negatives) ===")
    print(f"Evaluated on {evaluated} users")
    print(f"HR@{TOP_K}:   {hr:.4f}")
    print(f"NDCG@{TOP_K}: {ndcg:.4f}")
    print(f"\nOriginal NCF paper reports HR@10 ~0.68-0.72 on MovieLens datasets")
    print(f"(their setup, not a leak-free comparison to ours -- see caveat above)")

    return hr, ndcg


if __name__ == "__main__":
    run_paper_protocol_eval()