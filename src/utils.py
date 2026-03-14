"""
utils.py — save/load models, timing decorator, directory setup.
"""

import time
import functools
from pathlib import Path

import joblib

from src.config import MODELS_DIR, FIGURES_DIR, EDA_DIR, OUTPUTS_DIR


def ensure_dirs():
    """Create all output directories if they don't exist yet."""
    for d in [OUTPUTS_DIR, FIGURES_DIR, MODELS_DIR, EDA_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print("✓ Output directories ready")


def save_model(model, name: str):
    """Persist a fitted sklearn model to outputs/models/<name>.pkl"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    print(f"  saved → {path}")
    return path


def load_model(name: str):
    """Load a previously saved model."""
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}")
    return joblib.load(path)


def timer(func):
    """Decorator: print how long a function took."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"  ⏱  {func.__name__} took {elapsed:.1f}s")
        return result

    return wrapper


def save_fig(fig, name: str, subdir: str = "figures"):
    """
    Save a matplotlib figure to outputs/<subdir>/<n>.png.

    Does NOT close the figure — the caller decides when to close it.
    In notebooks: the figure stays alive so %matplotlib inline can render it.
    In scripts:   call plt.close(fig) after save_fig() to free memory.
    """
    out = OUTPUTS_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")
    return path


def print_section(title: str):
    """Nice little section header for console output."""
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")
