"""
Merge MovieLens (ratings/movies/links) with TMDB metadata into one clean
dataset, ready for both content-based and collaborative filtering.

Preprocessing steps done here (each one matters, skipping any of these
will bite you later during training):

1. Drop movies with no tmdbId link (can't fetch metadata -> can't use
   for content-based filtering)
2. Drop movies where the TMDB fetch failed/returned nothing
3. Fill missing overview/genres/cast with empty string/list instead of
   NaN (NaN breaks embedding models and string concatenation)
4. Deduplicate: MovieLens occasionally has near-duplicate title entries
   (different releases/rereleases) -- drop exact duplicate movieIds
5. Filter out movies with very few ratings (e.g. < 5) -- collaborative
   filtering can't learn a meaningful signal from 1-2 data points, and
   they just add noise/sparsity to the user-item matrix
6. Normalize text fields (lowercase, strip whitespace) for the genre/
   keyword/cast fields that'll feed into embeddings later
7. Build one "combined_text" field per movie (overview + genres + cast +
   director + keywords) -- this is what gets embedded for content-based
   filtering, so it needs to be clean and consistent
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_loader import load_ratings, load_movies, load_links
from src.tmdb_client import fetch_all_metadata


MIN_RATINGS_PER_MOVIE = 5  # tune this based on dataset size


def build_merged_dataset(min_ratings: int = MIN_RATINGS_PER_MOVIE) -> pd.DataFrame:
    ratings = load_ratings()
    movies = load_movies()
    links = load_links()

    # --- Step 5: filter sparse movies BEFORE hitting TMDB (saves API calls) ---
    rating_counts = ratings.groupby("movieId").size().rename("num_ratings")
    movies = movies.merge(rating_counts, on="movieId", how="left")
    movies["num_ratings"] = movies["num_ratings"].fillna(0).astype(int)
    movies = movies[movies["num_ratings"] >= min_ratings].copy()
    print(f"After min-ratings filter (>= {min_ratings}): {len(movies)} movies")

    # --- Step 1: join with links, drop movies with no tmdbId ---
    movies = movies.merge(links[["movieId", "tmdbId", "imdbId"]], on="movieId", how="inner")
    print(f"After dropping movies with no TMDB link: {len(movies)} movies")

    # --- Step 4: dedupe on movieId just in case ---
    movies = movies.drop_duplicates(subset=["movieId"])

    # --- Fetch TMDB metadata for all remaining movies ---
    tmdb_ids = movies["tmdbId"].tolist()
    tmdb_cache = fetch_all_metadata(tmdb_ids)

    # --- Step 2: attach metadata, drop movies TMDB fetch failed for ---
    def get_field(tmdb_id, field, default):
        record = tmdb_cache.get(tmdb_id)
        return record[field] if record and record.get(field) is not None else default

    movies["overview"] = movies["tmdbId"].apply(lambda t: get_field(t, "overview", ""))
    movies["tmdb_genres"] = movies["tmdbId"].apply(lambda t: get_field(t, "genres", []))
    movies["cast"] = movies["tmdbId"].apply(lambda t: get_field(t, "cast", []))
    movies["director"] = movies["tmdbId"].apply(lambda t: get_field(t, "director", ""))
    movies["keywords"] = movies["tmdbId"].apply(lambda t: get_field(t, "keywords", []))
    movies["poster_url"] = movies["tmdbId"].apply(lambda t: get_field(t, "poster_url", None))
    movies["vote_average"] = movies["tmdbId"].apply(lambda t: get_field(t, "vote_average", 0.0))
    movies["popularity"] = movies["tmdbId"].apply(lambda t: get_field(t, "popularity", 0.0))

    # drop rows where TMDB genuinely had nothing (fetch failed entirely)
    had_tmdb_data = movies["tmdbId"].apply(lambda t: t in tmdb_cache)
    movies = movies[had_tmdb_data].copy()
    print(f"After dropping failed TMDB fetches: {len(movies)} movies")

    # --- Step 3 + 6: normalize / fill missing text fields ---
    movies["overview"] = movies["overview"].fillna("").str.strip()
    movies["director"] = movies["director"].fillna("").str.strip()

    # prefer TMDB's genre list if present, fall back to MovieLens genres
    movies["final_genres"] = movies.apply(
        lambda r: r["tmdb_genres"] if r["tmdb_genres"] else r["genres_list"], axis=1
    )

    # --- Step 7: build the combined text field for embeddings later ---
    def combine_text(row):
        genres_text = " ".join(row["final_genres"])
        parts = [
            row["overview"],
            # repeat genres 3x so genre signal isn't drowned out by
            # longer overview/cast text during embedding
            (genres_text + " ") * 3,
            " ".join(row["cast"]),
            row["director"],
            " ".join(row["keywords"]),
        ]
        return " ".join(p for p in parts if p).lower().strip()

    movies["combined_text"] = movies.apply(combine_text, axis=1)

    # keep only movies that ended up with SOME text to embed
    movies = movies[movies["combined_text"].str.len() > 0].copy()
    print(f"Final dataset: {len(movies)} movies")

    final_cols = [
        "movieId", "tmdbId", "imdbId", "title_clean", "year",
        "final_genres", "overview", "cast", "director", "keywords",
        "combined_text", "poster_url", "vote_average", "popularity",
        "num_ratings",
    ]
    result = movies[final_cols].rename(columns={"title_clean": "title", "final_genres": "genres"})
    return result.reset_index(drop=True)


def clean_ratings_for_cf(ratings: pd.DataFrame, valid_movie_ids: set,
                          min_ratings_per_user: int = 5) -> pd.DataFrame:
    """
    Prep the ratings matrix specifically for collaborative filtering.
    Two extra things matter here beyond the movie-level filtering above:

    1. Restrict ratings to movies that survived the merge above (no point
       training CF on movies you have no metadata/poster for anyway)
    2. Filter out users with very few ratings -- a user with 1-2 ratings
       gives the model almost nothing to learn their taste from, and just
       adds sparsity/noise to the user-item matrix, same logic as the
       movie-side filtering
    """
    df = ratings[ratings["movieId"].isin(valid_movie_ids)].copy()

    user_counts = df.groupby("userId").size()
    valid_users = user_counts[user_counts >= min_ratings_per_user].index
    df = df[df["userId"].isin(valid_users)].copy()

    print(f"Ratings after cleaning: {len(df)} rows, "
          f"{df['userId'].nunique()} users, {df['movieId'].nunique()} movies")
    return df


if __name__ == "__main__":
    merged = build_merged_dataset()
    merged.to_parquet(config.MERGED_DATASET_PATH, index=False)
    print(f"Saved merged dataset -> {config.MERGED_DATASET_PATH}")
    print(merged.head())

    # also produce the cleaned ratings matrix ready for CF training
    ratings = load_ratings()
    clean_ratings = clean_ratings_for_cf(ratings, set(merged["movieId"]))
    clean_ratings_path = config.DATA_PROCESSED / "ratings_clean.parquet"
    clean_ratings.to_parquet(clean_ratings_path, index=False)
    print(f"Saved cleaned ratings -> {clean_ratings_path}")