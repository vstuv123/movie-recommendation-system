"""
Recommendation endpoints:
- GET  /recommendations/{user_id}          -> for known users (has rating history)
- POST /recommendations/cold-start          -> for new users, given liked movies
- GET  /movies/{movie_id}/similar           -> pure content-based "movies like this"
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Depends

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.dependencies import get_recommender
from app.schemas import (
    RecommendationResponse, MovieRecommendation,
    ColdStartRequest, SimilarMoviesResponse,
)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


def _row_to_recommendation(row) -> MovieRecommendation:
    """Shared conversion from a pandas row to the API response shape."""
    return MovieRecommendation(
        movieId=int(row["movieId"]),
        title=row["title"],
        genres=list(row["genres"]) if isinstance(row["genres"], (list, np.ndarray)) else [],
        vote_average=float(row["vote_average"]),
        num_ratings=int(row["num_ratings"]) if pd.notna(row.get("num_ratings")) else 0,
        poster_url=row.get("poster_url"),
        hybrid_score=float(row["hybrid_score"]),
        cf_predicted_rating=(
            float(row["cf_predicted_rating"])
            if pd.notna(row.get("cf_predicted_rating")) else None
        ),
        content_similarity=(
            float(row["content_similarity"])
            if pd.notna(row.get("content_similarity")) else None
        ),
    )


@router.get("/{user_id}", response_model=RecommendationResponse)
def get_recommendations(
    user_id: int,
    top_k: int = Query(default=10, ge=1, le=50),
    recommender=Depends(get_recommender),
):
    """
    Get recommendations for a user by their userId. Works for both known
    users (real CF + content hybrid scoring) and cold users with no rating
    history (automatically falls back to popularity-based recommendations).
    """
    try:
        recs_df = recommender.recommend(user_id, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {e}")

    is_cold_start = "cf_predicted_rating" not in recs_df.columns or recs_df["cf_predicted_rating"].isna().all()

    # add poster_url + num_ratings from the movies table (not included in
    # recommend()'s output by default)
    extra_cols = recommender.content.movies[["poster_url", "num_ratings"]]
    recs_df = recs_df.merge(extra_cols, left_on="movieId", right_index=True, how="left")

    return RecommendationResponse(
        user_id=user_id,
        is_cold_start=is_cold_start,
        recommendations=[_row_to_recommendation(row) for _, row in recs_df.iterrows()],
    )


@router.post("/cold-start", response_model=RecommendationResponse)
def get_cold_start_recommendations(
    request: ColdStartRequest,
    recommender=Depends(get_recommender),
):
    """
    For a brand-new user (e.g. during onboarding, "pick a few movies you
    like") -- pass movieIds instead of relying on rating history. Uses
    userId=-1 as a placeholder since there's no real account yet.
    """
    invalid_ids = [
        mid for mid in request.liked_movie_ids
        if mid not in recommender.content.id_to_idx
    ]
    if invalid_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown movieIds (not in catalog): {invalid_ids}"
        )

    try:
        recs_df = recommender.recommend(
            user_id=-1,  # placeholder, this user has no CF embedding
            top_k=request.top_k,
            liked_movie_ids=request.liked_movie_ids,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    extra_cols = recommender.content.movies[["poster_url", "num_ratings"]]
    recs_df = recs_df.merge(extra_cols, left_on="movieId", right_index=True, how="left")

    return RecommendationResponse(
        user_id=-1,
        is_cold_start=True,
        recommendations=[_row_to_recommendation(row) for _, row in recs_df.iterrows()],
    )


similar_router = APIRouter(prefix="/movies", tags=["movies"])


@similar_router.get("/{movie_id}/similar", response_model=SimilarMoviesResponse)
def get_similar_movies(
    movie_id: int,
    top_k: int = Query(default=10, ge=1, le=50),
    genre_boost: float = Query(default=0.15, ge=0.0, le=1.0),
    recommender=Depends(get_recommender),
):
    """
    Pure content-based "movies like this one" -- doesn't need a user at
    all, works for any movie in the catalog. Good for a movie detail page.
    """
    if movie_id not in recommender.content.id_to_idx:
        raise HTTPException(status_code=404, detail=f"movieId {movie_id} not found")

    similar_df = recommender.content.similar_to(movie_id, top_k=top_k, genre_boost=genre_boost)
    similar_df["hybrid_score"] = similar_df["similarity"]  # reuse the same response shape

    posters_and_ratings = recommender.content.movies[["poster_url", "num_ratings"]]
    similar_df = similar_df.merge(posters_and_ratings, left_on="movieId", right_index=True, how="left")

    movie_title = recommender.content.movies.loc[movie_id, "title"]

    return SimilarMoviesResponse(
        movie_id=movie_id,
        title=movie_title,
        similar_movies=[_row_to_recommendation(row) for _, row in similar_df.iterrows()],
    )