"""
Movie browsing endpoints -- search by title, get popular movies, get a
single movie's details. Useful for onboarding flows (search + pick liked
movies), a homepage "trending" row, and movie detail pages.
"""
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query, Depends

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.dependencies import get_recommender
from app.schemas import MovieSearchResult

router = APIRouter(prefix="/movies", tags=["movies"])


@router.get("/search", response_model=list[MovieSearchResult])
def search_movies(
    q: str = Query(..., min_length=1, description="Search query (matches movie title)"),
    limit: int = Query(default=20, ge=1, le=100),
    recommender=Depends(get_recommender),
):
    """Simple case-insensitive substring search over movie titles."""
    movies = recommender.content.movies
    matches = movies[movies["title"].str.contains(q, case=False, na=False, regex=False)]
    matches = matches.sort_values("num_ratings", ascending=False).head(limit)

    return [
        MovieSearchResult(
            movieId=int(movie_id),
            title=row["title"],
            genres=list(row["genres"]),
            year=int(row["year"]) if "year" in row and row["year"] == row["year"] else None,
            vote_average=float(row["vote_average"]),
            num_ratings=int(row["num_ratings"]),
            poster_url=row.get("poster_url"),
        )
        for movie_id, row in matches.iterrows()
    ]


@router.get("/popular", response_model=list[MovieSearchResult])
def get_popular_movies(
    limit: int = Query(default=20, ge=1, le=100),
    recommender=Depends(get_recommender),
):
    """Trending/popular movies -- good for a homepage row that doesn't
    depend on any specific user."""
    popular_df = recommender._popularity_recommendations(top_k=limit)

    return [
        MovieSearchResult(
            movieId=int(row["movieId"]),
            title=row["title"],
            genres=list(row["genres"]),
            year=None,
            vote_average=float(row["vote_average"]),
            num_ratings=int(row["num_ratings"]),
            poster_url=row.get("poster_url"),
        )
        for _, row in popular_df.iterrows()
    ]


@router.get("/{movie_id}", response_model=MovieSearchResult)
def get_movie(movie_id: int, recommender=Depends(get_recommender)):
    """Single movie's details, e.g. for a movie detail page."""
    movies = recommender.content.movies
    if movie_id not in movies.index:
        raise HTTPException(status_code=404, detail=f"movieId {movie_id} not found")

    row = movies.loc[movie_id]
    return MovieSearchResult(
        movieId=movie_id,
        title=row["title"],
        genres=list(row["genres"]),
        year=int(row["year"]) if "year" in row and row["year"] == row["year"] else None,
        vote_average=float(row["vote_average"]),
        num_ratings=int(row["num_ratings"]),
        poster_url=row.get("poster_url"),
    )