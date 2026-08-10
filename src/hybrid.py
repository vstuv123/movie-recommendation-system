"""
Hybrid recommender: combines the NCF collaborative filtering model with
content-based embeddings.

Why hybrid instead of picking one:
- NCF learns real community taste patterns, but can't score a movie or
  user it never saw during training (cold-start problem)
- Content-based works for ANY movie (even brand new ones with zero
  ratings) since it only needs metadata, but has no idea about actual
  user behavior patterns
- Blending them covers each one's blind spot

Scoring formula for a known user + known movie:
    final_score = alpha * (cf_predicted_rating / 5.0) + (1 - alpha) * content_score

Where content_score = how similar a candidate movie is to the movies
this user already rated highly (their "taste profile" in embedding space).

Cold-start handling:
- New user (no ratings in training data): fall back to pure content-based,
  using whatever movies they've explicitly liked as the taste profile
- New movie (not in the trained embedding index): can't be scored by NCF
  at all, so it can only appear via content-based recommendations
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.content_based import ContentRecommender
from src.collaborative import NeuMF


class HybridRecommender:
    def __init__(self, alpha: float = 0.4):
        """
        alpha: weight given to the collaborative filtering score.
        alpha=1.0 -> pure CF, alpha=0.0 -> pure content-based.
        0.4 is a reasonable starting point (favor CF slightly, since it
        captures real behavior, but still let content pull its weight).
        Tune this based on which one your eval metrics favor.
        """
        self.alpha = alpha
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        import os
        import gdown
        
        # Create directories on the server if they don't exist
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        os.makedirs(config.DATA_PROCESSED, exist_ok=True)

        # Download the heavy 176 MB ratings data file if it's missing
        ratings_file_path = config.DATA_PROCESSED / "ratings_clean.parquet"
        if not ratings_file_path.exists():
            print("📥 ratings_clean.parquet missing. Downloading from Google Drive...")
            ratings_file_id = "1mN6vJ7sCjVcHtCsNgA3lAtwhGzIZ6K92"
            gdown.download(id=ratings_file_id, output=str(ratings_file_path), quiet=False)

        print("Loading content-based recommender...")
        self.content = ContentRecommender()

        print("Loading NCF model...")
        self._load_ncf_model()

        print("Loading ratings for user taste profiles...")
        self.ratings = pd.read_parquet(ratings_file_path)

    def _load_ncf_model(self):

        import gdown
        pt_file_path = config.MODELS_DIR / "ncf_model.pt"
        mapping_file_path = config.MODELS_DIR / "ncf_id_mappings.npz"

        # Download PyTorch weights if missing
        if not pt_file_path.exists():
            print("📥 ncf_model.pt missing. Downloading from Google Drive...")
            pt_file_id = "1H_6LGkRwVJp60GUVxZPTVd-zH8fYNIAD"
            gdown.download(id=pt_file_id, output=str(pt_file_path), quiet=False)

        # Download index mapping tracking files if missing
        if not mapping_file_path.exists():
            print("📥 ncf_id_mappings.npz missing. Downloading from Google Drive...")
            mapping_file_id = "1ofcf6bzo_L2ihCLIL7QCC4C8Mg5JMvOA"
            gdown.download(id=mapping_file_id, output=str(mapping_file_path), quiet=False)

        # Updated original lines to read from the explicit file paths
        checkpoint = torch.load(pt_file_path, weights_only=False, map_location=self.device)
        self.ncf_model = NeuMF(
            checkpoint["num_users"],
            checkpoint["num_movies"],
            embed_dim=checkpoint["embed_dim"],
            mlp_layers=checkpoint["mlp_layers"],
            global_mean=checkpoint.get("global_mean", 3.5),
        ).to(self.device)
        self.ncf_model.load_state_dict(checkpoint["model_state_dict"])
        self.ncf_model.eval()

        # Updated to read from the explicit mapping path
        id_maps = np.load(mapping_file_path)
        user_ids = id_maps["user_ids"]
        movie_ids = id_maps["movie_ids"]
        self.user_id_to_idx = {u: i for i, u in enumerate(user_ids)}
        self.movie_id_to_idx = {m: i for i, m in enumerate(movie_ids)}
        self.ncf_movie_ids = movie_ids  # keep for index -> movieId lookups

    def _user_taste_profile(self, user_id: int, liked_movie_ids: list[int] = None,
                             top_n_liked: int = 20) -> np.ndarray:
        """
        Build a single embedding vector representing this user's taste,
        as a rating-weighted average of embeddings for movies they rated
        highly. Falls back to explicit liked_movie_ids for cold-start users.
        """
        if liked_movie_ids:
            valid_ids = [m for m in liked_movie_ids if m in self.content.id_to_idx]
            if not valid_ids:
                raise ValueError("None of the provided liked_movie_ids are in the embedding index")
            idxs = [self.content.id_to_idx[m] for m in valid_ids]
            return self.content.embeddings[idxs].mean(axis=0)

        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        if len(user_ratings) == 0:
            raise ValueError(
                f"userId {user_id} has no ratings and no liked_movie_ids provided -- "
                f"can't build a taste profile for a fully cold user"
            )

        top_rated = user_ratings.nlargest(top_n_liked, "rating")
        weights, vectors = [], []
        for _, row in top_rated.iterrows():
            if row["movieId"] in self.content.id_to_idx:
                idx = self.content.id_to_idx[row["movieId"]]
                vectors.append(self.content.embeddings[idx])
                weights.append(row["rating"])

        if not vectors:
            raise ValueError(f"None of userId {user_id}'s rated movies are in the embedding index")

        weights = np.array(weights)
        vectors = np.array(vectors)
        profile = (vectors * weights[:, None]).sum(axis=0) / weights.sum()
        return profile / np.linalg.norm(profile)  # renormalize to unit length

    def _cf_predict_all(self, user_id: int, chunk_size: int = 10000) -> dict:
        """
        Predict CF rating for EVERY movie in the NCF vocabulary for this
        user (one batched forward pass, chunked to keep memory sane).
        Used so CF can propose its own candidates instead of only
        re-ranking movies that content-based similarity already picked --
        otherwise a movie CF would love but that isn't a content-neighbor
        of the user's taste profile could never surface at all.
        """
        if user_id not in self.user_id_to_idx:
            return {}

        user_idx = self.user_id_to_idx[user_id]
        num_movies = len(self.ncf_movie_ids)
        all_preds = np.empty(num_movies, dtype=np.float32)

        with torch.no_grad():
            for start in range(0, num_movies, chunk_size):
                end = min(start + chunk_size, num_movies)
                m_idx = torch.arange(start, end, dtype=torch.long, device=self.device)
                u_idx = torch.full((end - start,), user_idx, dtype=torch.long, device=self.device)
                all_preds[start:end] = self.ncf_model(u_idx, m_idx).cpu().numpy()

        return dict(zip(self.ncf_movie_ids, all_preds))

    def _popularity_recommendations(self, top_k: int, exclude_ids: set = None) -> pd.DataFrame:
        """
        Cold-start fallback for a totally new user with zero ratings and
        no liked_movie_ids given -- there's no taste signal to build a
        profile from, so nearest-neighbor search in embedding space isn't
        possible yet. Fall back to "what's popular and well-rated overall",
        same thing Netflix shows you before it knows anything about you.

        Uses a Bayesian-average-style score so a movie with 5 ratings all
        5-star doesn't outrank a movie with 50,000 ratings averaging 4.3 --
        raw vote_average alone is unreliable for low-rating-count movies.
        """
        movies = self.content.movies.copy()
        if exclude_ids:
            movies = movies[~movies.index.isin(exclude_ids)]

        C = movies["vote_average"].mean()          # prior: overall average rating
        m = movies["num_ratings"].quantile(0.80)    # minimum ratings threshold to be "trustworthy"

        movies["bayesian_score"] = (
            (movies["num_ratings"] / (movies["num_ratings"] + m)) * movies["vote_average"]
            + (m / (movies["num_ratings"] + m)) * C
        )

        top = movies.nlargest(top_k, "bayesian_score")
        result = top[["title", "genres", "vote_average", "num_ratings"]].copy()
        result["hybrid_score"] = top["bayesian_score"] / 10.0  # scale to comparable 0-1 range
        result["cf_predicted_rating"] = np.nan
        result["content_similarity"] = np.nan
        return result.reset_index()

    def recommend(self, user_id: int, top_k: int = 10,
                   candidate_pool_size: int = 300,
                   liked_movie_ids: list[int] = None,
                   exclude_ids: set = None) -> pd.DataFrame:
        """
        Generate top_k hybrid recommendations for a user.

        liked_movie_ids: optional, for cold-start users with no training
        history -- pass a few movieIds they've explicitly liked instead.

        exclude_ids: optional override for which movies to exclude as
        candidates (defaults to "everything this user already rated").
        Mainly used by the evaluation script to do leave-one-out testing.
        """
        # These are two SEPARATE questions that shouldn't be tied together:
        # 1. Can CF score this user at all? -> depends only on whether they
        #    have a trained embedding (were in the NCF training set)
        # 2. Where does their taste profile come from? -> liked_movie_ids
        #    if given, otherwise their rating history
        # A user can be known to CF AND have liked_movie_ids passed in --
        # CF should still be used for them in that case.
        is_known_to_cf = user_id in self.user_id_to_idx
        has_ratings = (self.ratings["userId"] == user_id).any()

        # Fully cold user: no ratings history AND no explicit likes given.
        # No taste signal exists at all -- fall back to popularity.
        if not has_ratings and not liked_movie_ids:
            print(f"userId {user_id} has no ratings and no liked_movie_ids -- "
                  f"falling back to popularity-based recommendations")
            return self._popularity_recommendations(top_k, exclude_ids)

        # Step 1: get a taste profile vector for candidate generation
        taste_profile = self._user_taste_profile(user_id, liked_movie_ids)

        # Step 2: generate candidates from BOTH signals, not just content --
        # otherwise a movie CF would rate highly but that isn't a content-
        # neighbor of the taste profile could never enter the pool at all,
        # regardless of alpha.
        sims_to_profile = self.content.embeddings @ taste_profile
        content_candidate_idx = np.argpartition(sims_to_profile, -candidate_pool_size)[-candidate_pool_size:]
        content_candidate_ids = set(self.content.movie_ids[content_candidate_idx].tolist())

        cf_all_preds = self._cf_predict_all(user_id) if is_known_to_cf else {}
        if cf_all_preds:
            cf_top_ids = set(
                sorted(cf_all_preds, key=cf_all_preds.get, reverse=True)[:candidate_pool_size]
            )
        else:
            cf_top_ids = set()

        candidate_movie_ids = np.array(list(content_candidate_ids | cf_top_ids))

        # exclude movies the user has already rated (or an override set,
        # used by the eval script for leave-one-out testing) -- based on
        # has_ratings, independent of whether liked_movie_ids was also given
        if exclude_ids is not None:
            already_rated = exclude_ids
        elif has_ratings:
            already_rated = set(self.ratings.loc[self.ratings["userId"] == user_id, "movieId"])
        else:
            already_rated = set()

        keep_mask = np.array([mid not in already_rated for mid in candidate_movie_ids])
        candidate_movie_ids = candidate_movie_ids[keep_mask]

        # content score: cosine sim of each candidate to the taste profile
        content_idx_lookup = np.array([self.content.id_to_idx[mid] for mid in candidate_movie_ids])
        content_scores = self.content.embeddings[content_idx_lookup] @ taste_profile

        # CF score: pull from the precomputed dict (nan if this candidate
        # wasn't in the NCF vocabulary, e.g. too few ratings during training)
        if is_known_to_cf and cf_all_preds:
            cf_scores = np.array([cf_all_preds.get(mid, np.nan) for mid in candidate_movie_ids])
        else:
            cf_scores = np.full(len(candidate_movie_ids), np.nan)

        # Step 4: blend. Where CF has no score (cold-start movie/user),
        # fall back to pure content score for that candidate.
        content_norm = (content_scores + 1) / 2  # cosine sim [-1,1] -> [0,1]
        cf_norm = cf_scores / 5.0                 # rating [0.5,5] -> roughly [0,1]

        final_scores = np.where(
            np.isnan(cf_norm),
            content_norm,  # cold-start fallback: pure content
            self.alpha * cf_norm + (1 - self.alpha) * content_norm,
        )

        order = np.argsort(-final_scores)[:top_k]
        result_ids = candidate_movie_ids[order]

        result = self.content.movies.loc[result_ids, ["title", "genres", "vote_average"]].copy()
        result["hybrid_score"] = final_scores[order]
        result["cf_predicted_rating"] = cf_scores[order]
        result["content_similarity"] = content_scores[order]
        return result.reset_index()


if __name__ == "__main__":
    hybrid = HybridRecommender(alpha=0.6)

    sample_user_id = hybrid.ratings["userId"].iloc[0]
    print(f"\nHybrid recommendations for userId {sample_user_id}:")
    print(hybrid.recommend(sample_user_id, top_k=10))