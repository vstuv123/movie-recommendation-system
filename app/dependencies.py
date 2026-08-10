"""
Holds the loaded HybridRecommender as app-wide shared state.

IMPORTANT: loading the recommender (embeddings + NCF model + ratings) is
expensive (several seconds, GB-scale memory). This must happen ONCE at
app startup, not per-request -- see main.py's lifespan handler, which
calls load_recommender() before the app starts accepting traffic.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.hybrid import HybridRecommender

_recommender: HybridRecommender | None = None


def load_recommender(alpha: float = 0.4):
    """Call once at startup. alpha=0.4 is the empirically best value we
    found via the eval sweep in src/evaluate.py."""
    global _recommender
    _recommender = HybridRecommender(alpha=alpha)
    return _recommender


def get_recommender() -> HybridRecommender:
    """FastAPI dependency -- raises clearly if called before startup ran."""
    if _recommender is None:
        raise RuntimeError(
            "Recommender not loaded yet. load_recommender() must run in "
            "the app's startup/lifespan handler before requests are served."
        )
    return _recommender