"""
Unified Streamlit Frontend & Model Runner.
Runs the recommender natively inside the Streamlit app (no separate FastAPI
backend needed). Model weights and data files are downloaded from Google
Drive on first launch if they aren't already present locally.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import os
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

sys.path.append(str(Path(__file__).resolve().parent))
from src.hybrid import HybridRecommender

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

recommender = None

# =====================================================================
# 🧠 MEMORY-CACHED INITIALIZATION (boots model natively, once per session)
# =====================================================================
@st.cache_resource
def initialize_native_recommendation_system():
    """
    Downloads any missing files, then instantiates the full hybrid
    recommendation engine. Cached so this only runs once per app instance,
    not on every rerun/interaction.
    """

    with st.spinner("🧠 Booting recommendation neural layers..."):
        recommender_instance = HybridRecommender(alpha=0.4)

    return recommender_instance


try:
    recommender = initialize_native_recommendation_system()
    st.sidebar.success("🟢 AI Recommendation Engine Operational")
except Exception as e:
    st.sidebar.error("🔴 Engine Initialization Failed")
    st.error(f"Initialization crashed: {e}")
    st.info(
        "💡 Check that GDRIVE_FILE_IDS at the top of this file are filled in, "
        "and that each Google Drive file is shared as 'Anyone with the link'."
    )

# =====================================================================
# 🎨 SHARED INTERFACE RENDERING UTILITIES
# =====================================================================
def enrich_with_poster_and_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    recommend()/similar_to()/_popularity_recommendations() don't always
    include poster_url / num_ratings in their output columns -- merge them
    in from the full movies table so the UI always has them, regardless of
    which method produced this dataframe.
    """
    if df is None or df.empty:
        return df
    extra_cols = recommender.content.movies[["poster_url", "num_ratings"]]
    merged = df.merge(extra_cols, left_on="movieId", right_index=True, how="left",
                       suffixes=("", "_dup"))
    return merged.drop(columns=[c for c in merged.columns if c.endswith("_dup")], errors="ignore")


def render_movie_grid(movies_df, columns=5, key_prefix="movie", score_label="Match"):
    """Renders a dataframe of movies into a grid of poster cards."""
    if movies_df is None or movies_df.empty:
        st.info("No data available to display in grid.")
        return

    cols = st.columns(columns)

    for idx, (_, row) in enumerate(movies_df.iterrows()):
        with cols[idx % columns]:
            poster = row.get("poster_url") if "poster_url" in row and pd.notna(row["poster_url"]) else None
            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.markdown("🎬 *(no poster)*")

            st.markdown(f"**{row['title']}**")

            genres_list = row.get("genres", [])
            genres = ", ".join(genres_list[:3]) if isinstance(genres_list, (list, np.ndarray)) else str(genres_list)
            st.caption(genres)

            vote_avg = row.get("vote_average", 0)
            num_ratings = int(row.get("num_ratings", 0)) if pd.notna(row.get("num_ratings")) else 0
            st.caption(f"⭐ {vote_avg:.1f}  ·  {num_ratings:,} ratings")

            if "hybrid_score" in row and pd.notna(row["hybrid_score"]):
                st.progress(min(max(float(row["hybrid_score"]), 0.0), 1.0))
                st.caption(f"{score_label}: {row['hybrid_score']:.2f}")

            if st.button("Similar movies", key=f"{key_prefix}_{row.get('movieId', idx)}_{idx}"):
                st.session_state["similar_to_id"] = row.get("movieId")
                st.session_state["similar_to_title"] = row["title"]
                st.rerun()


# =====================================================================
# 🎬 UI MAIN WORKFLOW
# =====================================================================
st.title("🎬 Movie Recommender Portal")
st.caption("Hybrid Engine: Neural Collaborative Filtering + Content embeddings running natively.")

mode = st.sidebar.radio(
    "Navigation Modes",
    ["Popular / Trending", "Recommendations for a User", "New User (Cold Start)", "Search"],
)

# Global "similar movies" view -- triggered from any grid's button, shown
# regardless of which mode is currently selected
if recommender and "similar_to_id" in st.session_state:
    st.subheader(f"Movies similar to: {st.session_state['similar_to_title']}")
    movie_id = st.session_state["similar_to_id"]

    if movie_id in recommender.content.id_to_idx:
        similar_df = recommender.content.similar_to(movie_id, top_k=10)
        similar_df["hybrid_score"] = similar_df["similarity"]
        similar_df = enrich_with_poster_and_ratings(similar_df)
        render_movie_grid(similar_df, key_prefix="similar", score_label="Content match")
    else:
        st.warning("This movie isn't in the embedding index.")

    if st.button("Clear View"):
        del st.session_state["similar_to_id"]
        del st.session_state["similar_to_title"]
        st.rerun()
    st.divider()


if mode == "Popular / Trending":
    if recommender:
        st.subheader("Popular Right Now (Bayesian Metrics)")
        limit = st.slider("How many movies?", 5, 30, 15)
        popular_df = recommender._popularity_recommendations(top_k=limit)
        popular_df = enrich_with_poster_and_ratings(popular_df)
        render_movie_grid(popular_df, key_prefix="popular", score_label="Popularity Score")
    else:
        st.warning("Please resolve initialization errors to load data modules.")


elif mode == "Recommendations for a User":
    if recommender:
        st.subheader("Predict Existing User Taste Matrix")
        user_id = st.number_input("Enter target User ID:", min_value=1, value=1, step=1)
        top_k = st.slider("Total recommendations output:", 5, 30, 10)

        if st.button("Compute Predictions Pipeline", type="primary"):
            try:
                recs_df = recommender.recommend(user_id, top_k=top_k)
                recs_df = enrich_with_poster_and_ratings(recs_df)

                is_cold = "cf_predicted_rating" not in recs_df.columns or recs_df["cf_predicted_rating"].isna().all()
                if is_cold:
                    st.info("User has no baseline history. Displaying high-scoring trending items.")

                label = "Popularity score" if is_cold else "Match"
                render_movie_grid(recs_df, key_prefix="user_rec", score_label=label)
            except Exception as err:
                st.error(f"Failed running recommendation vector arrays: {err}")
    else:
        st.warning("Engine is currently offline.")


elif mode == "New User (Cold Start)":
    st.subheader("Onboarding Preferences Layout")
    st.caption("Search for a few movies you like -- we'll recommend similar ones using content-based matching (no rating history needed).")

    if recommender:
        search_query = st.text_input("Search for a movie to add")
        if search_query:
            movies = recommender.content.movies
            matches = movies[movies["title"].str.contains(search_query, case=False, na=False, regex=False)]
            matches = matches.sort_values("num_ratings", ascending=False).head(10)

            for movie_id, row in matches.iterrows():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{row['title']}** ({row.get('year', 'N/A')}) -- {', '.join(row['genres'][:3])}")
                with col2:
                    if st.button("Add", key=f"add_{movie_id}"):
                        liked = st.session_state.setdefault("liked_movies", [])
                        if movie_id not in [m["movieId"] for m in liked]:
                            liked.append({"movieId": movie_id, "title": row["title"]})
                        st.rerun()

        liked = st.session_state.get("liked_movies", [])
        if liked:
            st.write("**Your picks:**")
            for movie in liked:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"- {movie['title']}")
                with col2:
                    if st.button("Remove", key=f"remove_{movie['movieId']}"):
                        st.session_state["liked_movies"] = [
                            m for m in liked if m["movieId"] != movie["movieId"]
                        ]
                        st.rerun()

            if st.button("Get recommendations based on these", type="primary"):
                liked_ids = [m["movieId"] for m in liked]
                try:
                    recs_df = recommender.recommend(
                        user_id=-1,  # placeholder, no real CF embedding for this user
                        top_k=12,
                        liked_movie_ids=liked_ids,
                    )
                    recs_df = enrich_with_poster_and_ratings(recs_df)
                    st.subheader("Recommended for you")
                    render_movie_grid(recs_df, key_prefix="coldstart", score_label="Match")
                except ValueError as err:
                    st.error(str(err))
        else:
            st.info("Search and add a few movies above to get started.")
    else:
        st.warning("Engine is currently offline.")


elif mode == "Search":
    if recommender:
        st.subheader("Search the catalog")
        query = st.text_input("Movie title")
        if query:
            movies = recommender.content.movies
            matches = movies[movies["title"].str.contains(query, case=False, na=False, regex=False)]
            matches = matches.sort_values("num_ratings", ascending=False).head(24).reset_index()
            render_movie_grid(matches, key_prefix="search")
    else:
        st.warning("Engine is currently offline.")