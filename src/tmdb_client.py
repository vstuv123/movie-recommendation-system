"""
TMDB API client. Fetches metadata for a movie by tmdbId and caches results
to disk (JSONL) so re-running the pipeline doesn't re-hit the API for
movies you've already fetched.
"""
import json
import time
import sys
from pathlib import Path
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def _load_cache() -> dict:
    """Load already-fetched TMDB records, keyed by tmdbId."""
    cache = {}
    if config.TMDB_CACHE_PATH.exists():
        with open(config.TMDB_CACHE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    cache[record["tmdb_id"]] = record
    return cache


import threading
_cache_write_lock = threading.Lock()


def _append_to_cache(record: dict):
    # lock needed since multiple threads will call this concurrently now
    with _cache_write_lock:
        with open(config.TMDB_CACHE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def fetch_movie_metadata(tmdb_id: int) -> dict | None:
    """
    Fetch one movie's metadata + credits (cast/director) from TMDB.
    Returns None if the movie isn't found or the request fails.
    """
    if not config.TMDB_API_KEY:
        raise RuntimeError(
            "TMDB_API_KEY not set. Get a free key at "
            "https://www.themoviedb.org/settings/api and set it as an env var."
        )

    url = f"{config.TMDB_BASE_URL}/movie/{tmdb_id}"
    params = {
        "api_key": config.TMDB_API_KEY,
        "append_to_response": "credits,keywords",  # get cast/crew + keywords in ONE call
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except requests.RequestException:
        return None

    # Pull top 5 cast members and the director specifically
    cast = [c["name"] for c in data.get("credits", {}).get("cast", [])[:5]]
    crew = data.get("credits", {}).get("crew", [])
    director = next((c["name"] for c in crew if c["job"] == "Director"), None)
    keywords = [k["name"] for k in data.get("keywords", {}).get("keywords", [])]

    poster_path = data.get("poster_path")
    poster_url = (
        f"{config.TMDB_IMAGE_BASE}{config.TMDB_POSTER_SIZE}{poster_path}"
        if poster_path else None
    )

    record = {
        "tmdb_id": tmdb_id,
        "title": data.get("title"),
        "overview": data.get("overview", ""),
        "genres": [g["name"] for g in data.get("genres", [])],
        "cast": cast,
        "director": director,
        "keywords": keywords,
        "poster_url": poster_url,
        "release_date": data.get("release_date"),
        "runtime": data.get("runtime"),
        "vote_average": data.get("vote_average"),
        "vote_count": data.get("vote_count"),
        "popularity": data.get("popularity"),
        "original_language": data.get("original_language"),
    }
    return record


def fetch_all_metadata(tmdb_ids: list[int], verbose: bool = True,
                        max_workers: int = 20) -> dict:
    """
    Fetch metadata for a list of tmdbIds IN PARALLEL using a thread pool,
    using the on-disk cache to skip ones we already have. Returns dict of
    {tmdb_id: record}.

    Why threads instead of a plain loop: each requests.get() call spends
    almost all its time waiting on the network round-trip, not on CPU. A
    serial loop wastes that wait time doing nothing. Threads let many
    requests be "in flight" waiting on the network at once, so you're
    bottlenecked by TMDB's actual rate limit (~50 req/sec) instead of
    your own round-trip latency. 20 workers is a safe default -- bump to
    30-40 if you're not seeing 429 (rate limit) errors.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache = _load_cache()
    to_fetch = [tid for tid in tmdb_ids if tid not in cache]

    if verbose:
        print(f"{len(cache)} already cached, fetching {len(to_fetch)} new movies from TMDB...")

    if not to_fetch:
        return cache

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {executor.submit(fetch_movie_metadata, tid): tid for tid in to_fetch}
        for future in as_completed(future_to_id):
            tmdb_id = future_to_id[future]
            try:
                record = future.result()
            except Exception:
                record = None
            if record is not None:
                _append_to_cache(record)
                cache[tmdb_id] = record
            completed += 1
            if verbose and completed % 200 == 0:
                print(f"  fetched {completed}/{len(to_fetch)}")

    return cache


if __name__ == "__main__":
    # quick smoke test — Toy Story is tmdbId 862
    record = fetch_movie_metadata(862)
    print(json.dumps(record, indent=2))