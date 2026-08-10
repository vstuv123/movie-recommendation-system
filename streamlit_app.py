"""
Unified Streamlit Frontend & Model Runner
Bypasses the need for an external FastAPI backend by running the machine learning model locally in-container.
"""
import os
import sys
import gdown
import torch
import pandas as pd
import streamlit as st
from pathlib import Path

# Setup system environment paths
sys.path.append(str(Path(__file__).resolve().parent))
import config
from src.hybrid import HybridRecommender

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# Initialize the variable as None globally so it never triggers a NameError
recommender = None

# =====================================================================
# 🧠 MEMORY-CACHED INITIALIZATION (Boots model natively)
# =====================================================================
@st.cache_resource
def initialize_native_recommendation_system():
    """
    Instantiates the complete Hybrid Recommendation Engine.
    All background downloading is handled safely inside the class initialization.
    """
    with st.spinner("🧠 Booting recommendation neural layers..."):
        # The class constructor itself will now download any missing .pt or .parquet files!
        recommender_instance = HybridRecommender(alpha=0.4)
        
    return recommender_instance

# Safe system boot up
try:
    recommender = initialize_native_recommendation_system()
    st.sidebar.success("🟢 AI Recommendation Engine Operational")
except Exception as e:
    st.sidebar.error("🔴 Engine Initialization Failed")
    st.error(f"Initialization crashed: {e}")
    st.info("💡 Make sure your clean Parquet data files are committed to GitHub inside your 'data/processed/' folder!")

# =====================================================================
# 🎨 SHARED INTERFACE RENDERING UTILITIES
# =====================================================================
def render_movie_grid(movies_df, columns=5, key_prefix="movie", score_label="Match"):
    """Renders our dataframe results cleanly into visual thumbnail slots."""
    if movies_df is None or movies_df.empty:
        st.info("No data available to display in grid.")
        return
        
    cols = st.columns(columns)
    
    # Process dataframe rows dynamically
    for idx, (_, row) in enumerate(movies_df.iterrows()):
        with cols[idx % columns]:
            poster = row.get("poster_url") if "poster_url" in row and pd.notna(row["poster_url"]) else None
            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.markdown("🎬 *(no poster)*")

            st.markdown(f"**{row['title']}**")
            
            # Format genres safe checking
            genres_list = row.get("genres", [])
            genres = ", ".join(genres_list[:3]) if isinstance(genres_list, list) else str(genres_list)
            st.caption(genres)

            vote_avg = row.get("vote_average", 0)
            num_ratings = int(row.get("num_ratings", 0)) if pd.notna(row.get("num_ratings")) else 0
            st.caption(f"⭐ {vote_avg:.1f}  ·  {num_ratings:,} ratings")

            if "hybrid_score" in row and pd.notna(row["hybrid_score"]):
                st.progress(min(max(float(row["hybrid_score"]), 0.0), 1.0))
                st.caption(f"{score_label}: {row['hybrid_score']:.2f}")

            # Safe callback placeholder for similar movies browsing
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
    ["Popular / Trending", "Recommendations for a User", "New User (Cold Start)"],
)

# Global view intercept for similar movies navigation handling
if recommender and "similar_to_id" in st.session_state:
    st.subheader(f"Movies similar to: {st.session_state['similar_to_title']}")
    movies_idx = recommender.content.movies
    if st.session_state["similar_to_id"] in movies_idx.index:
        st.info("Displaying core vector properties...")
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
                
                extra_cols = recommender.content.movies[["poster_url", "num_ratings"]]
                recs_df = recs_df.merge(extra_cols, left_on="movieId", right_index=True, how="left")
                
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
    st.write("Native content-based matching pipeline goes here.")
