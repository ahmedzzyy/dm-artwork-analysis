"""
config.py — central config for paths, constants, model params.
Change things here; nowhere else.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
EDA_DIR = OUTPUTS_DIR / "eda"

ARTWORKS_CSV = DATA_DIR / "Artworks.csv"
ARTISTS_CSV = DATA_DIR / "Artists.csv"

# ── Dataset columns we care about ─────────────────────────────────────────
# After merging artworks + artists
ARTWORK_COLS = [
    "Title",
    "Artist",
    "ConstituentID",
    "ArtistBio",
    "Nationality",
    "BeginDate",
    "EndDate",
    "Gender",
    "Date",
    "Medium",
    "Dimensions",
    "Department",
    "DateAcquired",
    "URL",
]

# ── Target for classification ──────────────────────────────────────────────
TARGET_COL = "Department"

# Departments to keep (drop tiny ones that would wreck class balance)
# Adjust after first EDA run if needed
MIN_CLASS_SAMPLES = 100

# ── Feature engineering knobs ─────────────────────────────────────────────
MEDIUM_MAX_FEATURES = 100  # top N terms from Medium TF-IDF
DECADE_BIN_SIZE = 10  # years per decade bucket

# ── Model hyperparams ─────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None  # grow full trees, let min_samples prune

SVM_C = 1.0
SVM_KERNEL = "rbf"

LR_MAX_ITER = 1000

# ── Clustering knobs ──────────────────────────────────────────────────────
KMEANS_K_RANGE = range(2, 13)  # elbow method search range
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 5
PCA_N_COMPONENTS = 2

# ── Plot style ─────────────────────────────────────────────────────────────
PLOT_STYLE = "seaborn-v0_8-whitegrid"
PLOT_PALETTE = "tab10"
FIGSIZE_STD = (10, 6)
FIGSIZE_WIDE = (14, 6)
FIGSIZE_SQ = (8, 8)
DPI = 150
