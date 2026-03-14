"""
visualizer.py — shared plot utilities callable from both main.py and notebooks.
Heavy lifting is done in eda.py / classifier.py / clusterer.py;
this module has the "convenience combos" and any plots that don't fit elsewhere.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import PLOT_STYLE, FIGSIZE_STD, FIGSIZE_WIDE, TARGET_COL
from src.utils import save_fig

plt.style.use(PLOT_STYLE)


def class_balance_before_after(
    df_raw: pd.DataFrame, df_clean: pd.DataFrame
) -> plt.Figure:
    """
    Side-by-side bar charts showing class distribution before and after
    dropping tiny classes. Good 'data cleaning' slide for the notebook.
    """
    raw_counts = df_raw[TARGET_COL].value_counts().sort_values()
    clean_counts = df_clean[TARGET_COL].value_counts().sort_values()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    ax1.barh(raw_counts.index, raw_counts.values, color="#4C72B0")
    ax1.set_title("Raw dataset", fontweight="bold")
    ax1.set_xlabel("Count")

    ax2.barh(clean_counts.index, clean_counts.values, color="#55A868")
    ax2.set_title("After cleaning", fontweight="bold")
    ax2.set_xlabel("Count")

    fig.suptitle("Department Distribution — Before & After Cleaning", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "class_balance_before_after", subdir="figures")
    return fig


def feature_matrix_sample(clf_df: pd.DataFrame, n: int = 500) -> plt.Figure:
    """
    Heatmap of a random sample of the feature matrix.
    Great for showing 'what the model actually sees'.
    Clips extreme values for readability.
    """
    non_target = [c for c in clf_df.columns if c != "target"]
    sample = clf_df[non_target].sample(min(n, len(clf_df)), random_state=42)
    sample_clipped = sample.clip(
        lower=sample.quantile(0.01), upper=sample.quantile(0.99), axis=1
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        sample_clipped.T,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"shrink": 0.5},
    )
    ax.set_xlabel(f"{n} sample artworks →")
    ax.set_ylabel("Features →")
    ax.set_title("Feature Matrix Sample (clipped)", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "feature_matrix_sample", subdir="figures")
    return fig


def learning_curve_plot(model, X_train, y_train) -> plt.Figure:
    """
    Train/validation accuracy vs training set size.
    Reveals over/underfitting at a glance.
    """
    from sklearn.model_selection import learning_curve

    sizes, train_scores, val_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=3,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1,
    )

    t_mean, t_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    v_mean, v_std = val_scores.mean(axis=1), val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    ax.plot(sizes, t_mean, "o-", color="#4C72B0", label="Train accuracy")
    ax.fill_between(sizes, t_mean - t_std, t_mean + t_std, alpha=0.15, color="#4C72B0")
    ax.plot(sizes, v_mean, "s-", color="#C44E52", label="Val accuracy")
    ax.fill_between(sizes, v_mean - v_std, v_mean + v_std, alpha=0.15, color="#C44E52")

    ax.set_xlabel("Training set size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    save_fig(fig, "learning_curve", subdir="figures")
    return fig


def pca_explained_variance(
    X_scaled: np.ndarray, max_components: int = 50
) -> plt.Figure:
    """Cumulative explained variance by PCA component number."""
    from sklearn.decomposition import PCA

    n = min(max_components, X_scaled.shape[1])
    pca = PCA(n_components=n, random_state=42)
    pca.fit(X_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    ax.plot(range(1, n + 1), cumvar, "b-o", markersize=4)
    ax.axhline(0.80, linestyle="--", color="gray", label="80% threshold")
    ax.axhline(0.90, linestyle=":", color="gray", label="90% threshold")
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA — Cumulative Explained Variance", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    save_fig(fig, "pca_variance", subdir="figures")
    return fig


def summary_dashboard(
    df_clean: pd.DataFrame, best_result: dict, km_summary: pd.DataFrame
) -> plt.Figure:
    """
    4-panel summary figure for the final notebook / report cover page:
      [0,0] department bar chart
      [0,1] model accuracy comparison
      [1,0] creation decade histogram
      [1,1] cluster sizes pie chart
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MoMA Art Mining — Summary Dashboard", fontsize=14, fontweight="bold")

    # Panel 1 — Department distribution
    ax = axes[0, 0]
    counts = df_clean[TARGET_COL].value_counts()
    ax.barh(counts.index, counts.values, color=sns.color_palette("tab10", len(counts)))
    ax.set_title("Artworks by Department")
    ax.set_xlabel("Count")

    # Panel 2 — Model accuracy (from best_result context)
    ax = axes[0, 1]
    if "all_results" in best_result:
        labels = [r["label"] for r in best_result["all_results"]]
        accs = [r["accuracy"] for r in best_result["all_results"]]
    else:
        labels = [best_result.get("label", "Best model")]
        accs = [best_result.get("accuracy", 0)]
    bars = ax.bar(labels, accs, color=["#4C72B0", "#55A868", "#C44E52"][: len(labels)])
    ax.set_ylim(0, 1.0)
    ax.set_title("Model Accuracy")
    ax.set_ylabel("Accuracy")
    for b, a in zip(bars, accs):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.01,
            f"{a:.3f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    # Panel 3 — Creation decade histogram
    ax = axes[1, 0]
    dec = df_clean.dropna(subset=["creation_decade"])["creation_decade"].astype(int)
    dec = dec[(dec >= 1860) & (dec <= 2020)]
    ax.hist(
        dec, bins=range(1860, 2025, 10), color="#4C72B0", edgecolor="none", alpha=0.8
    )
    ax.set_title("Artworks by Creation Decade")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Count")

    # Panel 4 — Cluster sizes
    ax = axes[1, 1]
    valid_clusters = km_summary[km_summary["cluster"] != "noise"]
    sizes = valid_clusters["size"].values
    clabels = [f"Cluster {c}" for c in valid_clusters["cluster"].values]
    ax.pie(
        sizes,
        labels=clabels,
        autopct="%1.0f%%",
        colors=sns.color_palette("tab10", len(sizes)),
        startangle=140,
        wedgeprops=dict(edgecolor="white"),
    )
    ax.set_title("KMeans Cluster Sizes")

    fig.tight_layout()
    save_fig(fig, "summary_dashboard", subdir="figures")
    return fig
