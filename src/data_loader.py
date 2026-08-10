"""
Load raw MovieLens files into pandas DataFrames.
"""
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def load_ratings() -> pd.DataFrame:
    """
    Columns: userId, movieId, rating, timestamp
    rating is 0.5 - 5.0 in half-star increments.
    """
    if not config.MOVIELENS_RATINGS.exists():
        raise FileNotFoundError(
            f"{config.MOVIELENS_RATINGS} not found. Download MovieLens from "
            f"https://grouplens.org/datasets/movielens/ and unzip into data/raw/"
        )
    df = pd.read_csv(config.MOVIELENS_RATINGS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def load_movies() -> pd.DataFrame:
    """
    Columns: movieId, title (includes year in parens), genres (pipe-separated)
    """
    df = pd.read_csv(config.MOVIELENS_MOVIES)
    # Extract year from title into its own column, e.g. "Toy Story (1995)" -> 1995
    df["year"] = df["title"].str.extract(r"\((\d{4})\)$").astype("Int64")
    df["title_clean"] = df["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)
    df["genres_list"] = df["genres"].apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )
    return df


def load_links() -> pd.DataFrame:
    """
    Columns: movieId, imdbId, tmdbId
    This is the bridge table between MovieLens IDs and TMDB IDs.
    """
    df = pd.read_csv(config.MOVIELENS_LINKS)
    # tmdbId can be missing (float NaN) for a handful of movies — drop those,
    # we can't fetch metadata for them anyway
    df = df.dropna(subset=["tmdbId"])
    df["tmdbId"] = df["tmdbId"].astype(int)
    return df


def load_tags() -> pd.DataFrame:
    """
    Columns: userId, movieId, tag, timestamp
    Free-text tags users applied to movies. Optional, useful as extra
    content-based signal (treat like weak keywords).
    """
    if not config.MOVIELENS_TAGS.exists():
        return pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])
    return pd.read_csv(config.MOVIELENS_TAGS)


if __name__ == "__main__":
    ratings = load_ratings()
    movies = load_movies()
    links = load_links()
    print(f"Ratings: {ratings.shape}")
    print(f"Movies:  {movies.shape}")
    print(f"Links:   {links.shape}")
    print(movies.head())