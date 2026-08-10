"""
Content-based filtering.

Pipeline:
1. Load movies_merged.parquet (has combined_text per movie from preprocessing)
2. Embed combined_text using a pretrained sentence-transformer -> one dense
   vector per movie (this IS the "deep learning" part -- no training loop
   needed since the model is pretrained, we're just doing inference)
3. Build a similarity index (cosine similarity) so we can quickly find
   "movies most similar to movie X"
4. Save embeddings to disk so we never have to recompute them
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

EMBEDDINGS_PATH = config.MODELS_DIR / "content_embeddings.npy"
EMBEDDINGS_MOVIE_IDS_PATH = config.MODELS_DIR / "content_embeddings_movie_ids.npy"

# all-MiniLM-L6-v2: fast, 384-dim, good quality-to-speed tradeoff.
# Runs fine on CPU too (embedding is cheap compared to training), but will
# auto-use GPU if available since sentence-transformers checks torch.cuda.
MODEL_NAME = "all-MiniLM-L6-v2"


def generate_embeddings(force_recompute: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (embeddings, movie_ids) where embeddings[i] is the vector for
    movie_ids[i]. Cached to disk after first run.
    """
    if EMBEDDINGS_PATH.exists() and not force_recompute:
        print("Loading cached embeddings...")
        return np.load(EMBEDDINGS_PATH), np.load(EMBEDDINGS_MOVIE_IDS_PATH)

    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Encoding on device: {device}")

    movies = pd.read_parquet(config.MERGED_DATASET_PATH)
    model = SentenceTransformer(MODEL_NAME, device=device)

    texts = movies["combined_text"].tolist()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # pre-normalize so cosine sim = dot product
    )

    movie_ids = movies["movieId"].to_numpy()

    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(EMBEDDINGS_MOVIE_IDS_PATH, movie_ids)
    print(f"Saved {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")

    return embeddings, movie_ids


class ContentRecommender:
    """
    Nearest-neighbor lookup over movie embeddings for content-based
    recommendations. Since embeddings are pre-normalized, cosine
    similarity == dot product, so this is just a matrix multiply.
    """

    def __init__(self):
        self.embeddings, self.movie_ids = generate_embeddings()
        self.id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}
        self.movies = pd.read_parquet(config.MERGED_DATASET_PATH).set_index("movieId")

    def similar_to(self, movie_id: int, top_k: int = 10,
                    genre_boost: float = 0.15) -> pd.DataFrame:
        """
        Return top_k movies most similar to the given movie_id.

        genre_boost: how much to reward genre overlap on top of raw
        embedding similarity. 0 = pure embedding similarity (can let
        outliers like shared-keyword-but-wrong-genre movies sneak in).
        Set higher (e.g. 0.3) for stricter genre matching, 0 to disable.
        """
        if movie_id not in self.id_to_idx:
            raise ValueError(f"movieId {movie_id} not in embedding index")

        idx = self.id_to_idx[movie_id]
        query_vec = self.embeddings[idx]
        query_genres = set(self.movies.loc[movie_id, "genres"])

        # cosine similarity against every movie (dot product since normalized)
        sims = self.embeddings @ query_vec
        sims[idx] = -1  # exclude the movie itself

        # widen the candidate pool beyond top_k so genre re-ranking has
        # room to promote/demote before we cut down to top_k
        pool_size = min(top_k * 5, len(sims))
        pool_indices = np.argpartition(sims, -pool_size)[-pool_size:]

        # compute genre overlap (Jaccard) for each candidate in the pool
        pool_ids = self.movie_ids[pool_indices]
        genre_overlap = np.array([
            self._jaccard(query_genres, set(self.movies.loc[mid, "genres"]))
            for mid in pool_ids
        ])

        final_scores = sims[pool_indices] + genre_boost * genre_overlap
        order = np.argsort(-final_scores)[:top_k]
        top_indices = pool_indices[order]

        result_ids = self.movie_ids[top_indices]
        result = self.movies.loc[result_ids, ["title", "genres", "vote_average"]].copy()
        result["similarity"] = sims[top_indices]
        result["genre_overlap"] = genre_overlap[order]
        return result.reset_index()

    @staticmethod
    def _jaccard(set_a: set, set_b: set) -> float:
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    def similar_to_text(self, free_text_query: str, top_k: int = 10) -> pd.DataFrame:
        """
        Bonus: find movies matching an arbitrary text query, e.g.
        'time travel heist thriller with a twist ending'.
        Useful for a search bar in your frontend later.
        """
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(MODEL_NAME, device=device)
        query_vec = model.encode(free_text_query, normalize_embeddings=True)

        sims = self.embeddings @ query_vec
        top_indices = np.argpartition(sims, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-sims[top_indices])]

        result_ids = self.movie_ids[top_indices]
        result = self.movies.loc[result_ids, ["title", "genres", "vote_average"]].copy()
        result["similarity"] = sims[top_indices]
        return result.reset_index()


if __name__ == "__main__":
    recommender = ContentRecommender()

    sample_id = recommender.movie_ids[0]
    sample_title = recommender.movies.loc[sample_id, "title"]
    print(f"\nMovies similar to '{sample_title}':")
    print(recommender.similar_to(sample_id, top_k=10))