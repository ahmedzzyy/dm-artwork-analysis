"""
clusterer.py — unsupervised discovery of artwork groups.

Steps:
  1. Scale features
  2. Reduce to 2D with PCA (fast) and t-SNE (slow, pretty)
  3. Elbow + silhouette to find optimal k
  4. KMeans clustering
  5. DBSCAN clustering
  6. Human-readable cluster profiles
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

from src.config import (
    RANDOM_STATE,
    KMEANS_K_RANGE,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    PCA_N_COMPONENTS,
    PLOT_STYLE,
    FIGSIZE_STD,
    FIGSIZE_SQ,
    FIGSIZE_WIDE,
    TARGET_COL,
)
from src.utils import save_fig, save_model, timer, print_section

plt.style.use(PLOT_STYLE)


# ── Dimensionality reduction ───────────────────────────────────────────────


@timer
def reduce_pca(
    X: np.ndarray, n_components: int = PCA_N_COMPONENTS
) -> tuple[np.ndarray, PCA]:
    """PCA to n_components. Fast, used for elbow + silhouette too."""
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA {n_components}D: {explained:.1%} variance explained")
    return X_pca, pca


@timer
def reduce_tsne(X_pca: np.ndarray, perplexity: int = 30) -> np.ndarray:
    """
    t-SNE for final visualisation.
    Input should be PCA-reduced first (speeds things up enormously).
    """
    # Subsample for speed if dataset is huge
    N = len(X_pca)
    if N > 15_000:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(N, 15_000, replace=False)
        print(f"  t-SNE: subsampled {N:,} → 15,000")
    else:
        idx = np.arange(N)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=RANDOM_STATE,
        n_iter=1000,
    )
    X_tsne_sub = tsne.fit_transform(X_pca[idx])

    # Return both the embedding and the indices used
    return X_tsne_sub, idx


# ── Optimal k selection ────────────────────────────────────────────────────


@timer
def find_optimal_k(X_pca: np.ndarray) -> tuple[int, plt.Figure]:
    """
    Plot elbow (inertia) and silhouette scores for k in KMEANS_K_RANGE.
    Returns (suggested_k, figure).
    """
    inertias = []
    silhouettes = []
    ks = list(KMEANS_K_RANGE)

    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        labels = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        silhouettes.append(
            silhouette_score(X_pca, labels, sample_size=5000, random_state=RANDOM_STATE)
        )
        print(f"    k={k:2d}  inertia={km.inertia_:,.0f}  sil={silhouettes[-1]:.4f}")

    # Auto-suggest k = argmax silhouette
    best_k = ks[int(np.argmax(silhouettes))]
    print(f"  → Suggested k = {best_k} (highest silhouette)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    ax1.plot(ks, inertias, "bo-", linewidth=2)
    ax1.set_xlabel("Number of clusters k")
    ax1.set_ylabel("Inertia (within-cluster SS)")
    ax1.set_title("Elbow Method", fontweight="bold")
    ax1.axvline(best_k, linestyle="--", color="red", alpha=0.5, label=f"k={best_k}")
    ax1.legend()

    ax2.plot(ks, silhouettes, "gs-", linewidth=2)
    ax2.set_xlabel("Number of clusters k")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score", fontweight="bold")
    ax2.axvline(best_k, linestyle="--", color="red", alpha=0.5, label=f"k={best_k}")
    ax2.legend()

    fig.suptitle("Choosing Optimal Number of Clusters", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "elbow_silhouette", subdir="figures")
    return best_k, fig


# ── Clustering algorithms ──────────────────────────────────────────────────


@timer
def run_kmeans(X_pca: np.ndarray, k: int) -> tuple[np.ndarray, KMeans]:
    """Fit KMeans with the chosen k. Returns (labels, model)."""
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_pca)

    sil = silhouette_score(X_pca, labels, sample_size=5000, random_state=RANDOM_STATE)
    db = davies_bouldin_score(X_pca, labels)
    print(f"  KMeans k={k}: silhouette={sil:.4f}, Davies-Bouldin={db:.4f}")
    return labels, km


@timer
def run_dbscan(X_pca: np.ndarray) -> np.ndarray:
    """DBSCAN clustering. Noise points get label -1."""
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1)
    labels = db.fit_predict(X_pca)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(
        f"  DBSCAN: {n_clusters} clusters, {n_noise:,} noise points ({n_noise / len(labels):.1%})"
    )

    if n_clusters > 1:
        mask = labels != -1
        sil = silhouette_score(
            X_pca[mask], labels[mask], sample_size=5000, random_state=RANDOM_STATE
        )
        print(f"  DBSCAN silhouette (excl. noise): {sil:.4f}")
    return labels


# ── Cluster interpretation ────────────────────────────────────────────────


def analyze_clusters(
    df_clean: pd.DataFrame, labels: np.ndarray, method: str = "KMeans"
) -> pd.DataFrame:
    """
    Attach cluster labels to the original clean dataframe and summarise:
      - size of each cluster
      - top departments
      - top nationalities
      - median creation decade

    Returns a summary DataFrame (great for notebook display).
    """
    df2 = df_clean.copy().reset_index(drop=True)
    df2["cluster"] = labels

    rows = []
    for cluster_id in sorted(df2["cluster"].unique()):
        grp = df2[df2["cluster"] == cluster_id]
        size = len(grp)

        top_dept = (
            grp[TARGET_COL].value_counts().head(3).index.tolist()
            if TARGET_COL in grp.columns
            else []
        )

        nat_col = (
            "ArtistNationality" if "ArtistNationality" in grp.columns else "Nationality"
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

        label_str = "noise" if cluster_id == -1 else str(cluster_id)
        rows.append(
            {
                "cluster": label_str,
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
    X_2d: np.ndarray,
    labels: np.ndarray,
    title: str = "PCA Cluster Plot",
    true_dept_labels: pd.Series = None,
) -> plt.Figure:
    """Scatter plot of 2D PCA, colored by cluster."""
    unique = sorted(set(labels))
    palette = sns.color_palette("tab10", len(unique))
    color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(unique)}
    colors = [color_map[l] for l in labels]

    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.4, s=6, linewidths=0)

    # Legend (skip noise if too many clusters)
    if len(unique) <= 15:
        from matplotlib.patches import Patch

        handles = [
            Patch(color=color_map[l], label="noise" if l == -1 else f"Cluster {l}")
            for l in unique
        ]
        ax.legend(
            handles=handles,
            loc="upper right",
            fontsize=7,
            markerscale=2,
            framealpha=0.7,
        )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, title.lower().replace(" ", "_"), subdir="figures")
    return fig


def plot_tsne_clusters(
    X_tsne: np.ndarray, labels_sub: np.ndarray, title: str = "t-SNE Cluster Plot"
) -> plt.Figure:
    """Same as above but for the t-SNE embedding."""
    return plot_pca_clusters(X_tsne, labels_sub, title=title)


def plot_cluster_dept_breakdown(
    df_clean: pd.DataFrame, labels: np.ndarray, title: str = "Cluster Composition"
) -> plt.Figure:
    """Stacked bar: for each cluster, what % of artworks are in each department."""
    df2 = df_clean.copy().reset_index(drop=True)
    df2["cluster"] = labels
    df2 = df2[df2["cluster"] != -1]  # drop DBSCAN noise

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


@timer
def run_clustering(clust_df: pd.DataFrame, df_clean: pd.DataFrame) -> tuple[dict, dict]:
    """
    Full clustering pipeline.

    Parameters
    ----------
    clust_df  : feature matrix (no target col)
    df_clean  : original clean df (for interpretability)

    Returns
    -------
    results   : dict with labels and summary for each method
    plots     : dict of {name: figure}
    """
    print_section("Clustering")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clust_df.values)

    # Reduce to 50 dims first (faster silhouette etc.), then 2D for viz
    pca50, _ = reduce_pca(X_scaled, n_components=min(50, X_scaled.shape[1]))
    X_2d, pca2d = reduce_pca(X_scaled, n_components=2)

    plots = {}

    # ── KMeans ──────────────────────────────────────────────────────────
    print("\nFinding optimal k...")
    best_k, fig_elbow = find_optimal_k(pca50)
    plots["elbow"] = fig_elbow

    print(f"\nRunning KMeans (k={best_k})...")
    km_labels, km_model = run_kmeans(pca50, best_k)
    save_model(km_model, f"kmeans_k{best_k}")

    km_summary = analyze_clusters(df_clean, km_labels, method="KMeans")
    plots["pca_kmeans"] = plot_pca_clusters(
        X_2d, km_labels, title=f"PCA — KMeans k={best_k}"
    )
    plots["dept_kmeans"] = plot_cluster_dept_breakdown(
        df_clean, km_labels, title=f"KMeans k={best_k} — Department Composition"
    )

    # ── DBSCAN ──────────────────────────────────────────────────────────
    print("\nRunning DBSCAN...")
    db_labels = run_dbscan(X_2d)  # DBSCAN on 2D for meaningful eps
    db_summary = analyze_clusters(df_clean, db_labels, method="DBSCAN")
    plots["pca_dbscan"] = plot_pca_clusters(X_2d, db_labels, title="PCA — DBSCAN")

    # ── t-SNE (visual only) ──────────────────────────────────────────────
    print("\nRunning t-SNE (this takes a minute)...")
    X_tsne, tsne_idx = reduce_tsne(pca50)
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

    print(f"\n✓ Clustering done. {len(plots)} plots saved.")
    return results, plots
