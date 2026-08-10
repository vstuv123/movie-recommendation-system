"""
Central config for the movie recommender project.
Keep API keys out of git — load from environment variable, never hardcode.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # reads .env from project root automatically

# ---- Paths ----
BASE_DIR = Path(__file__).resolve().parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_CACHE = BASE_DIR / "data" / "cache"
MODELS_DIR = BASE_DIR / "models"

# ---- MovieLens ----
# Download the "ml-latest-small" (100k ratings, ~9k movies, good for dev/testing)
# or "ml-25m" (25M ratings, for a serious final model) from:
# https://grouplens.org/datasets/movielens/
# Unzip into data/raw/ so you have:
#   data/raw/ratings.csv
#   data/raw/movies.csv
#   data/raw/links.csv
#   data/raw/tags.csv
MOVIELENS_RATINGS = DATA_RAW / "ratings.csv"
MOVIELENS_MOVIES = DATA_RAW / "movies.csv"
MOVIELENS_LINKS = DATA_RAW / "links.csv"
MOVIELENS_TAGS = DATA_RAW / "tags.csv"

# ---- TMDB ----
# Get a free API key: https://www.themoviedb.org/settings/api
# Set it as an environment variable, don't hardcode it:
#   export TMDB_API_KEY="your_key_here"      (mac/linux)
#   setx TMDB_API_KEY "your_key_here"        (windows, restart terminal after)
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/"
TMDB_POSTER_SIZE = "w500"  # options: w92, w154, w185, w342, w500, w780, original

# TMDB free tier rate limit is generous (~50 req/sec) but be a good citizen
TMDB_REQUEST_DELAY_SEC = 0.05

# ---- Output ----
MERGED_DATASET_PATH = DATA_PROCESSED / "movies_merged.parquet"
TMDB_CACHE_PATH = DATA_CACHE / "tmdb_metadata.jsonl"

for _dir in [DATA_RAW, DATA_PROCESSED, DATA_CACHE, MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)