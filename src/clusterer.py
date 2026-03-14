"""
clusterer.py — unsupervised discovery of artwork groups.

Steps:
  1. Scale features
  2. PCA to 50D (fast distance computation) + 2D (visualisation)
  3. Elbow + silhouette to find optimal k  (k >= 3 enforced)
  4. KMeans clustering
  5. DBSCAN with auto-tuned eps
  6. t-SNE on the 2D PCA — tiny sample, low memory
  7. Human-readable cluster profiles
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

from src.config import (
    RANDOM_STATE,
    KMEANS_K_RANGE,
    PCA_N_COMPONENTS,
    PLOT_STYLE,
    FIGSIZE_STD,
    FIGSIZE_SQ,
    FIGSIZE_WIDE,
    TARGET_COL,
)
from src.utils import save_fig, save_model, timer, print_section

plt.style.use(PLOT_STYLE)

# Never suggest k=2 (trivial split) — require at least 3 clusters
KMEANS_K_MIN = 3


# ── Dimensionality reduction ───────────────────────────────────────────────


@timer
def reduce_pca(
    X: np.ndarray, n_components: int = PCA_N_COMPONENTS
) -> tuple[np.ndarray, PCA]:
    """PCA to n_components. Returns (X_reduced, fitted_pca)."""
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA {n_components}D: {explained:.1%} variance explained")
    return X_pca, pca


@timer
def reduce_tsne(
    X_2d: np.ndarray, max_samples: int = 8_000, perplexity: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """
    t-SNE for visualisation only.

    IMPORTANT: takes the *2D PCA* as input, not the 50D one.
    sklearn t-SNE allocates an N×N similarity matrix — passing 50D features
    at 15K samples uses ~11 GB and gets OOM-killed.
    2D input at 8K samples uses ~500 MB and runs in ~2 min.
    """
    N = len(X_2d)
    rng = np.random.default_rng(RANDOM_STATE)
    if N > max_samples:
        idx = rng.choice(N, max_samples, replace=False)
        print(f"  t-SNE: subsampled {N:,} → {max_samples:,} (memory guard)")
    else:
        idx = np.arange(N)

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(idx) // 4),  # perplexity < N/4
        random_state=RANDOM_STATE,
        max_iter=1000,
        init="pca",  # faster + more stable than random init
        learning_rate="auto",
    )
    X_tsne = tsne.fit_transform(X_2d[idx])
    return X_tsne, idx


# ── Optimal k ─────────────────────────────────────────────────────────────


@timer
def find_optimal_k(X_pca: np.ndarray) -> tuple[int, plt.Figure]:
    """
    Elbow + silhouette for k in KMEANS_K_RANGE.
    Enforces best_k >= KMEANS_K_MIN to avoid degenerate 2-cluster solutions.
    Returns (best_k, figure).
    """
    inertias = []
    silhouettes = []
    ks = list(KMEANS_K_RANGE)

    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        labels = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        sil = silhouette_score(
            X_pca, labels, sample_size=5_000, random_state=RANDOM_STATE
        )
        silhouettes.append(sil)
        print(f"    k={k:2d}  inertia={km.inertia_:,.0f}  sil={sil:.4f}")

    # Best k = highest silhouette, but never below KMEANS_K_MIN
    eligible = [(sil, k) for k, sil in zip(ks, silhouettes) if k >= KMEANS_K_MIN]
    best_k = max(eligible, key=lambda x: x[0])[1]
    print(f"  → Suggested k = {best_k} (best silhouette among k ≥ {KMEANS_K_MIN})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    ax1.plot(ks, inertias, "bo-", linewidth=2)
    ax1.set_xlabel("Number of clusters k")
    ax1.set_ylabel("Inertia (within-cluster SS)")
    ax1.set_title("Elbow Method", fontweight="bold")
    ax1.axvline(
        best_k, linestyle="--", color="red", alpha=0.6, label=f"chosen k={best_k}"
    )
    ax1.legend()

    ax2.plot(ks, silhouettes, "gs-", linewidth=2)
    ax2.set_xlabel("Number of clusters k")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score", fontweight="bold")
    ax2.axvline(
        best_k, linestyle="--", color="red", alpha=0.6, label=f"chosen k={best_k}"
    )
    ax2.axvline(
        KMEANS_K_MIN,
        linestyle=":",
        color="gray",
        alpha=0.5,
        label=f"min k={KMEANS_K_MIN}",
    )
    ax2.legend()

    fig.suptitle("Choosing Optimal Number of Clusters", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "elbow_silhouette", subdir="figures")
    return best_k, fig


# ── Clustering algorithms ──────────────────────────────────────────────────


@timer
def run_kmeans(X_pca: np.ndarray, k: int) -> tuple[np.ndarray, KMeans]:
    """Fit KMeans. Returns (labels, model)."""
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_pca)

    sil = silhouette_score(X_pca, labels, sample_size=5_000, random_state=RANDOM_STATE)
    db = davies_bouldin_score(X_pca, labels)
    print(f"  KMeans k={k}: silhouette={sil:.4f}, Davies-Bouldin={db:.4f}")
    return labels, km


def _auto_eps(X_2d: np.ndarray, min_samples: int) -> float:
    """
    Estimate DBSCAN eps from the kNN distance distribution.

    Uses the *median* of k-th nearest-neighbour distances + 20% headroom.
    Subsamples to 10K for speed. Enforces a hard floor of 1e-3.

    Why median instead of a low quantile:
      On dense datasets a low percentile (e.g. 5th) can hit exactly 0 when
      many points share the same PCA coordinate after rounding. The median
      is far more robust to that collapse.
    """
    N = len(X_2d)
    if N > 10_000:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(N, 10_000, replace=False)
        X_sample = X_2d[idx]
    else:
        X_sample = X_2d

    nbrs = NearestNeighbors(n_neighbors=min_samples, n_jobs=-1)
    nbrs.fit(X_sample)
    distances, _ = nbrs.kneighbors(X_sample)
    knn_dists = distances[:, -1]  # dist to k-th neighbour per point

    eps = float(np.median(knn_dists)) * 1.2  # median + 20% headroom
    eps = max(eps, 1e-3)  # hard floor — never 0

    print(
        f"  DBSCAN auto-eps: {eps:.4f}  "
        f"(median kNN={np.median(knn_dists):.4f}, k={min_samples})"
    )
    return eps


@timer
def run_dbscan(
    X_2d: np.ndarray, min_samples: int = 10, eps: float = None
) -> np.ndarray:
    """
    DBSCAN on the 2D PCA embedding.
    eps is auto-tuned from kNN distances if not supplied.
    Noise points get label -1.
    """
    if eps is None:
        eps = _auto_eps(X_2d, min_samples=min_samples)

    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X_2d)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(
        f"  DBSCAN: {n_clusters} clusters, {n_noise:,} noise points ({n_noise / len(labels):.1%})"
    )

    if n_clusters > 1:
        mask = labels != -1
        sil = silhouette_score(
            X_2d[mask], labels[mask], sample_size=5_000, random_state=RANDOM_STATE
        )
        print(f"  DBSCAN silhouette (excl. noise): {sil:.4f}")
    return labels


# ── Cluster interpretation ─────────────────────────────────────────────────


def analyze_clusters(
    df_clean: pd.DataFrame, labels: np.ndarray, method: str = "KMeans"
) -> pd.DataFrame:
    """
    Attach cluster labels to the original clean df and build a profile table:
      cluster | size | % | top departments | top nationalities | median decade
    """
    df2 = df_clean.copy().reset_index(drop=True)
    df2["cluster"] = labels

    nat_col = (
        "ArtistNationality" if "ArtistNationality" in df2.columns else "Nationality"
    )

    rows = []
    for cluster_id in sorted(df2["cluster"].unique()):
        grp = df2[df2["cluster"] == cluster_id]
        size = len(grp)

        top_dept = (
            grp[TARGET_COL].value_counts().head(3).index.tolist()
            if TARGET_COL in grp.columns
            else []
        )

        top_nat = (
            grp[nat_col].dropna().value_counts().head(3).index.tolist()
            if nat_col in grp.columns
            else []
        )

        med_decade = (
            grp["creation_decade"].median()
            if "creation_decade" in grp.columns
            else np.nan
        )

        rows.append(
            {
                "cluster": "noise" if cluster_id == -1 else str(cluster_id),
                "size": size,
                "pct": f"{size / len(df2):.1%}",
                "top_depts": ", ".join(top_dept) if top_dept else "—",
                "top_nats": ", ".join(top_nat) if top_nat else "—",
                "median_decade": int(med_decade) if pd.notna(med_decade) else "—",
            }
        )

    summary = pd.DataFrame(rows)
    print(f"\n{method} Cluster Summary:")
    print(summary.to_string(index=False))
    return summary


# ── Plotting ───────────────────────────────────────────────────────────────


def plot_pca_clusters(
    X_2d: np.ndarray, labels: np.ndarray, title: str = "PCA Cluster Plot"
) -> plt.Figure:
    """2D scatter coloured by cluster label."""
    unique = sorted(set(labels))
    palette = sns.color_palette("tab10", max(len(unique), 2))
    cmap = {lab: palette[i % len(palette)] for i, lab in enumerate(unique)}
    colors = [cmap[l] for l in labels]

    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.35, s=4, linewidths=0)

    if len(unique) <= 20:
        from matplotlib.patches import Patch

        handles = [
            Patch(color=cmap[l], label="noise" if l == -1 else f"Cluster {l}")
            for l in unique
        ]
        ax.legend(
            handles=handles,
            loc="upper right",
            fontsize=7,
            markerscale=3,
            framealpha=0.7,
        )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()

    fname = re.sub(r"[^\w]", "_", title.lower()).strip("_")
    save_fig(fig, fname, subdir="figures")
    return fig


def plot_tsne_clusters(
    X_tsne: np.ndarray, labels_sub: np.ndarray, title: str = "t-SNE"
) -> plt.Figure:
    return plot_pca_clusters(X_tsne, labels_sub, title=title)


def plot_cluster_dept_breakdown(
    df_clean: pd.DataFrame, labels: np.ndarray, title: str = "Cluster Composition"
) -> plt.Figure:
    """Stacked bar: department breakdown per cluster."""
    df2 = df_clean.copy().reset_index(drop=True)
    df2["cluster"] = labels
    df2 = df2[df2["cluster"] != -1]

    pivot = df2.groupby(["cluster", TARGET_COL]).size().unstack(fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot) * 1.5), 5))
    pivot_pct.plot(kind="bar", stacked=True, ax=ax, colormap="tab10", edgecolor="none")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Proportion")
    ax.set_title(title, fontweight="bold")
    ax.legend(
        title="Department", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    save_fig(fig, "cluster_dept_breakdown", subdir="figures")
    return fig


# ── Orchestrator ───────────────────────────────────────────────────────────

import re  # needed by plot_pca_clusters fname sanitiser above


@timer
def run_clustering(clust_df: pd.DataFrame, df_clean: pd.DataFrame) -> tuple[dict, dict]:
    """
    Full clustering pipeline.

    Memory budget
    -------------
    - PCA 50D: fine at any N
    - KMeans on 50D: fine
    - DBSCAN on 2D: fine
    - t-SNE: operates on 2D PCA, capped at 8K samples → ~500 MB peak

    Returns
    -------
    results : dict  {kmeans, dbscan, pca_2d, tsne}
    plots   : dict  {name: Figure}
    """
    print_section("Clustering")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clust_df.values)

    # Two PCA projections: 50D for clustering maths, 2D for visualisation
    X_pca50, _ = reduce_pca(X_scaled, n_components=min(50, X_scaled.shape[1]))
    X_2d, _ = reduce_pca(X_scaled, n_components=2)

    plots = {}

    # ── KMeans ──────────────────────────────────────────────────────────
    print("\nFinding optimal k...")
    best_k, fig_elbow = find_optimal_k(X_pca50)
    plots["elbow"] = fig_elbow

    print(f"\nRunning KMeans (k={best_k})...")
    km_labels, km_model = run_kmeans(X_pca50, best_k)
    save_model(km_model, f"kmeans_k{best_k}")

    km_summary = analyze_clusters(df_clean, km_labels, "KMeans")
    plots["pca_kmeans"] = plot_pca_clusters(
        X_2d, km_labels, title=f"PCA — KMeans k={best_k}"
    )
    plots["dept_kmeans"] = plot_cluster_dept_breakdown(
        df_clean, km_labels, title=f"KMeans k={best_k} — Department Composition"
    )

    # ── DBSCAN ──────────────────────────────────────────────────────────
    print("\nRunning DBSCAN (on 2D PCA, eps auto-tuned)...")
    db_labels = run_dbscan(X_2d)
    db_summary = analyze_clusters(df_clean, db_labels, "DBSCAN")
    plots["pca_dbscan"] = plot_pca_clusters(X_2d, db_labels, title="PCA — DBSCAN")

    # ── t-SNE ────────────────────────────────────────────────────────────
    # Feed 2D PCA (not 50D!) — this is the memory fix.
    # At 8K samples × 2 features sklearn t-SNE peaks at ~500 MB.
    print("\nRunning t-SNE on 2D PCA (capped at 8K samples)...")
    X_tsne, tsne_idx = reduce_tsne(X_2d, max_samples=8_000)
    km_labels_sub = km_labels[tsne_idx]
    plots["tsne_kmeans"] = plot_tsne_clusters(
        X_tsne, km_labels_sub, title=f"t-SNE — KMeans k={best_k}"
    )

    results = {
        "kmeans": {
            "labels": km_labels,
            "k": best_k,
            "model": km_model,
            "summary": km_summary,
        },
        "dbscan": {"labels": db_labels, "summary": db_summary},
        "pca_2d": X_2d,
        "tsne": {"embedding": X_tsne, "indices": tsne_idx},
    }

    print(f"\n✓ Clustering done — {len(plots)} plots saved.")
    return results, plots
