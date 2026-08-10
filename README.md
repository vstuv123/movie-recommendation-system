# 🎬 Movie Recommender System

A hybrid movie recommendation engine combining **content-based filtering**
(sentence-transformer embeddings over movie metadata) with **deep learning
collaborative filtering** (Neural Collaborative Filtering), trained on
real-world data and served through a FastAPI backend + Streamlit frontend.

---

## Overview

This system predicts what movies a user will enjoy by combining two
fundamentally different signals:

- **What the movie is about** — genre, plot, cast, director, keywords,
  captured as dense embeddings via a pretrained sentence-transformer
- **How real users actually behave** — a Neural Collaborative Filtering
  (NeuMF) model trained on 31.8M real ratings, learning taste patterns
  purely from user-movie interaction data

Neither signal alone is enough — content-based filtering doesn't know
anything about actual community taste, and collaborative filtering can't
say anything about a movie or user it's never seen. The hybrid layer blends
both, with automatic graceful fallback for brand-new users with no history.

---

## Key Results

| Metric | Result | Benchmark |
|---|---|---|
| **NCF Test RMSE** | **0.7643** | Netflix Prize baseline: 0.9525 · Netflix Prize winning solution: 0.8567 |
| **NCF Test MAE** | **0.5761** | (0.5–5.0 star scale) |
| **Hybrid Hit Rate@10** (leave-one-out) | **16.67%** (α=0.4) | vs. 3.67% pure content-based · 9.67% pure CF alone |
| **NCF-paper-protocol HR@10** | 0.504 | Original NCF paper (He et al. 2017): ~0.68–0.72 (different training objective, not directly comparable — see notes below) |

**The hybrid model beats both of its component models individually** —
this is the core empirical result validating the hybrid architecture,
not just an assumption baked in by design.

Our NCF model's rating-prediction accuracy **exceeds the Netflix Prize's
$1M-winning ensemble solution** (100+ blended models) on RMSE, on a
comparable (though not identical) rating-prediction task.

---

## Architecture

```
                        ┌─────────────────────┐
                        │   MovieLens 25M      │
                        │  (31.8M ratings)      │
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │   TMDB API           │
                        │ (genres, cast, plot,  │
                        │  posters, keywords)   │
                        └──────────┬───────────┘
                                   │
                          [ preprocessing ]
                                   │
              ┌────────────────────┴────────────────────┐
              │                                          │
    ┌─────────▼──────────┐                   ┌──────────▼───────────┐
    │  Content-Based       │                   │  Collaborative        │
    │  Filtering            │                   │  Filtering (NCF)      │
    │                        │                   │                        │
    │  sentence-transformer  │                   │  NeuMF: GMF + MLP      │
    │  embeddings (384-dim)  │                   │  branches + bias terms │
    │  + genre-aware          │                   │  trained on GPU        │
    │  re-ranking             │                   │                        │
    └─────────┬──────────┘                   └──────────┬───────────┘
              │                                          │
              └────────────────────┬────────────────────┘
                                   │
                        ┌──────────▼───────────┐
                        │    Hybrid Layer        │
                        │  alpha-weighted blend   │
                        │  + cold-start fallback  │
                        └──────────┬───────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                              │
          ┌─────────▼─────────┐         ┌─────────▼─────────┐
          │   FastAPI backend   │         │  Streamlit frontend │
          └────────────────────┘         └────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data | MovieLens 25M, TMDB API |
| Content embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Deep learning | PyTorch (CUDA, mixed precision) |
| Backend API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Data processing | pandas, NumPy, PyArrow (Parquet) |

---

## Dataset

| | |
|---|---|
| Movies | 43,216 (after filtering movies with insufficient ratings) |
| Ratings | 31,840,977 |
| Users | 200,948 |
| Rating scale | 0.5 – 5.0 stars (half-star increments) |
| Source | [MovieLens 25M](https://grouplens.org/datasets/movielens/) + [TMDB](https://www.themoviedb.org/) |

---

## Methodology

### 1. Content-Based Filtering
Each movie's overview, genres (weighted 3x for stronger signal), cast,
director, and keywords are concatenated into a single text blob and
encoded into a 384-dimensional embedding using a pretrained
sentence-transformer. Similarity between movies is cosine similarity
(a simple dot product, since embeddings are pre-normalized). Results are
re-ranked using genre-overlap (Jaccard similarity) to avoid
thematically-adjacent-but-genre-mismatched recommendations.

### 2. Collaborative Filtering — Neural Collaborative Filtering (NeuMF)
A two-branch deep learning architecture (He et al. 2017, adapted for
explicit rating regression):

- **GMF branch**: element-wise product of user/item embeddings (classic
  matrix factorization — linear taste-matching)
- **MLP branch**: concatenated embeddings through dense layers with
  BatchNorm + Dropout (captures non-linear interaction patterns)
- **Bias terms**: global + per-user + per-item bias (classic Netflix
  Prize SVD++ technique — frees the embeddings to focus purely on taste
  signal instead of baseline rating tendencies)

**Training**: 90/5/5 train/val/test split, full-GPU-resident tensors with
GPU-side batching (avoids CPU↔GPU transfer bottleneck), mixed precision
(`torch.amp`), gradient clipping, `ReduceLROnPlateau` scheduling, and
early stopping on validation RMSE — best checkpoint (not last epoch) used
for final evaluation. Trains in ~8 minutes on an RTX A2000 8GB.

### 3. Hybrid Layer
For a given user, candidates are generated from the **union** of
content-based nearest-neighbors and CF-predicted top-rated movies (so
neither signal is bottlenecked by the other during candidate generation).
Final ranking blends both:

```
final_score = alpha * (cf_predicted_rating / 5.0) + (1 - alpha) * content_similarity
```

`alpha = 0.4` was selected empirically via a leave-one-out evaluation
sweep (see Results above) — not an arbitrary default.

**Cold-start handling** covers three distinct cases:
- Known user (rating history) → full hybrid scoring
- New user with a few explicitly liked movies → pure content-based taste
  profile (genuine personalization, no CF needed)
- Fully cold user (nothing at all) → Bayesian-weighted popularity
  fallback (protects against low-rating-count movies looking artificially
  good)

---

## Project Structure

```
movie-recommender/
├── data/
│   ├── raw/                    # raw MovieLens CSVs
│   ├── processed/               # cleaned/merged parquet files
│   └── cache/                    # cached TMDB API responses
├── models/                       # trained model weights + embeddings
├── src/
│   ├── data_loader.py            # MovieLens loading
│   ├── tmdb_client.py            # TMDB fetching (threaded, cached)
│   ├── preprocessing.py          # merge + clean pipeline
│   ├── content_based.py          # embedding generation + similarity
│   ├── collaborative.py          # NeuMF model + GPU training loop
│   ├── hybrid.py                  # blended recommender + cold-start logic
│   ├── evaluate.py                # leave-one-out alpha sweep
│   └── evaluate_ncf_paper_protocol.py  # replicates He et al. 2017 protocol
├── app/                           # FastAPI backend
│   ├── main.py
│   ├── dependencies.py
│   ├── schemas.py
│   └── routes/
├── streamlit_app.py                # Streamlit frontend (standalone deployment)
├── config.py
└── requirements.txt
```

---

## Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get the data
- Download [MovieLens 25M](https://grouplens.org/datasets/movielens/) →
  unzip into `data/raw/`
- Get a free [TMDB API key](https://www.themoviedb.org/settings/api) →
  set as `TMDB_API_KEY` environment variable (or in a `.env` file)

### 3. Run the pipeline
```bash
python -m src.preprocessing          # merge MovieLens + TMDB, clean data
python -m src.content_based          # generate content embeddings
python -m src.collaborative          # train the NCF model (GPU recommended)
python -m src.evaluate               # (optional) sweep alpha values
```

### 4. Serve it

**Option A — FastAPI backend + Streamlit frontend (two processes):**
```bash
uvicorn app.main:app --port 8088
# in a second terminal:
streamlit run streamlit_app.py
```

**Option B — standalone Streamlit (no separate backend, for cloud deployment):**
Run the model natively inside Streamlit, downloading weights from Google
Drive on first launch. See `streamlit_app.py`'s `GDRIVE_FILE_IDS` config.

Interactive API docs available at `http://localhost:8088/docs` once the
backend is running.

---

## Notes on Evaluation Honesty

This project deliberately reports evaluation results with their caveats
rather than cherry-picking favorable comparisons:

- **RMSE/MAE** are directly comparable to Netflix Prize benchmarks (same
  metric, same task — rating prediction) and our model performs strongly.
- **Hit Rate@10 under the original NCF paper's protocol** (0.504) is
  lower than the paper's reported ~0.68–0.72. This is explained by a
  **training objective mismatch**: our model was trained via regression
  against explicit star ratings (MSE loss), not the paper's implicit
  feedback + negative-sampling ranking objective. It was never trained
  to directly distinguish real interactions from random negatives, so a
  same-numbers comparison to a model built specifically for that task
  isn't a fair apples-to-apples read.
- **Our own leave-one-out Hit Rate@10 sweep** (comparing our own
  content-only, CF-only, and hybrid variants against each other, all
  evaluated identically) is the most methodologically sound comparison in
  this project, and clearly shows the hybrid approach winning.

---

## Future Work

- Retrain NCF with a proper leave-one-out-by-timestamp split for a fully
  rigorous comparison against the original paper's protocol
- Implicit feedback signals (watch time, skip behavior) — not available
  in public datasets, would require production-scale telemetry
- Two-stage retrieval + ranking pipeline (candidate generation at scale
  via approximate nearest neighbors, e.g. FAISS) for larger catalogs
- Session/sequence-aware modeling (e.g. SASRec, BERT4Rec) for
  recency-sensitive recommendations

---

## Author

Built by Abdullah — BS Data Science, NUST SEECS, Islamabad.
