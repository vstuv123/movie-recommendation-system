"""
Evaluate the hybrid recommender using leave-one-out Hit Rate@K, and use
it to find the best alpha (CF vs content-based blend weight).

How leave-one-out evaluation works:
1. Pick a sample of users who have several highly-rated movies (>=4 stars)
2. For each user, HIDE one of their highly-rated movies (pretend they
   never rated it) -- this is the "held-out" ground truth item
3. Generate top-K recommendations for that user as normal, but explicitly
   allow the held-out movie to be a candidate again (exclude everything
   else they rated, but not this one)
4. Check: did the held-out movie show up in the top-K list?
5. Hit Rate@K = (fraction of users where the held-out movie appeared in
   their top-K) across all sampled users

This directly answers "if I hide something I know this user liked, does
the model recommend it back?" -- which is exactly what you want a
recommender to do in practice.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.hybrid import HybridRecommender

TOP_K = 10
NUM_EVAL_USERS = 300          # sample size -- more users = more reliable metric, but slower
MIN_RATING_FOR_HELD_OUT = 4.0  # only hide movies the user genuinely liked


def sample_eval_users(ratings: pd.DataFrame, known_user_ids: set, n: int) -> list[int]:
    """Pick users who have enough ratings to make a fair test, and who
    are in the NCF training set (so we're testing the actual hybrid,
    not just the cold-start fallback)."""
    candidates = ratings[ratings["userId"].isin(known_user_ids)]
    user_counts = candidates.groupby("userId").size()
    eligible_users = user_counts[user_counts >= 10].index.tolist()

    rng = np.random.default_rng(42)
    n = min(n, len(eligible_users))
    return rng.choice(eligible_users, size=n, replace=False).tolist()


def evaluate_alpha(hybrid: HybridRecommender, eval_users: list[int],
                    ratings: pd.DataFrame, alpha: float, top_k: int = TOP_K) -> float:
    hybrid.alpha = alpha
    hits = 0
    evaluated = 0

    for user_id in eval_users:
        user_ratings = ratings[ratings["userId"] == user_id]
        liked = user_ratings[user_ratings["rating"] >= MIN_RATING_FOR_HELD_OUT]
        if len(liked) == 0:
            continue

        # hold out one liked movie at random
        held_out_row = liked.sample(1, random_state=hash(user_id) % (2**31))
        held_out_movie_id = held_out_row["movieId"].values[0]

        # exclude everything the user rated EXCEPT the held-out movie,
        # so it's eligible to be recommended back
        all_rated = set(user_ratings["movieId"])
        exclude_ids = all_rated - {held_out_movie_id}

        try:
            recs = hybrid.recommend(user_id, top_k=top_k, exclude_ids=exclude_ids)
        except ValueError:
            continue  # e.g. movie not in embedding index, skip this user

        evaluated += 1
        if held_out_movie_id in recs["movieId"].values:
            hits += 1

    hit_rate = hits / evaluated if evaluated > 0 else 0.0
    print(f"  alpha={alpha:.1f} -> Hit Rate@{top_k}: {hit_rate:.4f} ({hits}/{evaluated} users)")
    return hit_rate


def run_alpha_sweep(alphas: list[float] = None):
    if alphas is None:
        alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    print("Loading hybrid recommender (this loads both models once)...")
    hybrid = HybridRecommender(alpha=0.6)
    ratings = hybrid.ratings

    print(f"\nSampling {NUM_EVAL_USERS} eval users...")
    eval_users = sample_eval_users(ratings, set(hybrid.user_id_to_idx.keys()), NUM_EVAL_USERS)
    print(f"Evaluating on {len(eval_users)} users\n")

    results = {}
    for alpha in alphas:
        hit_rate = evaluate_alpha(hybrid, eval_users, ratings, alpha)
        results[alpha] = hit_rate

    best_alpha = max(results, key=results.get)
    print(f"\n=== Results summary (Hit Rate@{TOP_K}) ===")
    for alpha, hr in results.items():
        marker = "  <-- best" if alpha == best_alpha else ""
        print(f"  alpha={alpha:.1f}: {hr:.4f}{marker}")

    print(f"\nBest alpha: {best_alpha}")
    print("Set this as the default alpha in HybridRecommender.__init__ for production use.")
    return results


if __name__ == "__main__":
    run_alpha_sweep()