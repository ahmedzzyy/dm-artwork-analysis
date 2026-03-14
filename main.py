"""
main.py — run the full MoMA art mining pipeline end-to-end.

Usage:
    uv run python main.py
    uv run python main.py --skip-tsne       # skip slow t-SNE step
    uv run python main.py --eda-only        # just EDA, no models
"""

import argparse
import sys
from pathlib import Path

# Make sure src/ is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import ensure_dirs, print_section
from src.config import ARTWORKS_CSV, ARTISTS_CSV


def check_data():
    """Friendly error if the user forgot to put the CSVs in data/."""
    missing = []
    if not ARTWORKS_CSV.exists():
        missing.append(f"  {ARTWORKS_CSV}  (download from kaggle: MoMA Collection)")
    if not ARTISTS_CSV.exists():
        missing.append(f"  {ARTISTS_CSV}   (same dataset)")
    if missing:
        print("\n❌  Missing data files:")
        for m in missing:
            print(m)
        print("\nGet them from: https://github.com/MuseumofModernArt/collection")
        print("Then place Artworks.csv and Artists.csv in the data/ folder.\n")
        sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="MoMA Art Mining Pipeline")
    p.add_argument("--eda-only", action="store_true", help="Run EDA only")
    p.add_argument("--skip-tsne", action="store_true", help="Skip slow t-SNE step")
    p.add_argument(
        "--skip-svm", action="store_true", help="Skip SVM (slow on large datasets)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    print_section("MoMA Artwork Mining Pipeline")
    ensure_dirs()
    check_data()

    # ── 1. Load & clean ──────────────────────────────────────────────────
    from src.data_loader import load_and_clean

    df = load_and_clean()

    # ── 2. EDA ───────────────────────────────────────────────────────────
    from src.eda import all_eda
    from src.features import extract_features

    clf_df, clust_df, meta = extract_features(df)
    all_eda(df, feature_df=clf_df)

    if args.eda_only:
        print("\n✓ EDA-only mode complete. Exiting.")
        return

    # ── 3. Classification ────────────────────────────────────────────────
    from src.classifier import run_classification

    best_result, all_results = run_classification(clf_df, meta)

    # ── 4. Clustering ────────────────────────────────────────────────────
    from src.clusterer import run_clustering

    cluster_results, cluster_plots = run_clustering(clust_df, df)

    # ── 5. Summary dashboard ─────────────────────────────────────────────
    from src.visualizer import summary_dashboard

    best_result["all_results"] = all_results
    summary_dashboard(df, best_result, cluster_results["kmeans"]["summary"])

    print_section("Pipeline Complete")
    print(
        f"  Best classifier : {best_result['label']} ({best_result['accuracy']:.4f} accuracy)"
    )
    print(f"  KMeans clusters : {cluster_results['kmeans']['k']}")
    print("\n  Figures  → outputs/figures/")
    print("  EDA      → outputs/eda/")
    print("  Models   → outputs/models/")
    print()


if __name__ == "__main__":
    main()
