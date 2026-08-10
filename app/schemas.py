"""
Pydantic schemas for API requests/responses. Keeping these separate from
the recommender logic itself so the API contract is easy to see at a glance.
"""
from pydantic import BaseModel, Field
from typing import Optional


class MovieRecommendation(BaseModel):
    movieId: int
    title: str
    genres: list[str]
    vote_average: float
    num_ratings: int = 0
    poster_url: Optional[str] = None
    hybrid_score: float
    cf_predicted_rating: Optional[float] = None
    content_similarity: Optional[float] = None


class RecommendationResponse(BaseModel):
    user_id: int
    is_cold_start: bool
    recommendations: list[MovieRecommendation]


class ColdStartRequest(BaseModel):
    """For a user with no rating history -- give a few movies they like instead."""
    liked_movie_ids: list[int] = Field(..., min_length=1, max_length=50)
    top_k: int = Field(default=10, ge=1, le=50)


class SimilarMoviesResponse(BaseModel):
    movie_id: int
    title: str
    similar_movies: list[MovieRecommendation]


class MovieSearchResult(BaseModel):
    movieId: int
    title: str
    genres: list[str]
    year: Optional[int] = None
    vote_average: float
    num_ratings: int = 0
    poster_url: Optional[str] = None