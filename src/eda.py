"""
eda.py — every exploratory plot for the MoMA dataset.
Each function returns a matplotlib Figure so notebooks can display it inline,
and also saves it to outputs/eda/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from src.config import (
    TARGET_COL,
    PLOT_STYLE,
    PLOT_PALETTE,
    FIGSIZE_STD,
    FIGSIZE_WIDE,
    FIGSIZE_SQ,
)
from src.utils import save_fig

plt.style.use(PLOT_STYLE)


# ── helpers ────────────────────────────────────────────────────────────────


def _add_value_labels(ax, fmt="{:.0f}", fontsize=9, rotation=0):
    """Slap count labels on top of every bar."""
    for patch in ax.patches:
        h = patch.get_height()
        if h > 0:
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                h + ax.get_ylim()[1] * 0.005,
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=fontsize,
                rotation=rotation,
            )


# ── Individual plot functions ──────────────────────────────────────────────


def department_distribution(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of artwork counts per department (our classification target)."""
    counts = df[TARGET_COL].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    bars = ax.barh(
        counts.index, counts.values, color=sns.color_palette(PLOT_PALETTE, len(counts))
    )
    ax.set_xlabel("Number of Artworks")
    ax.set_title("MoMA Collection — Artworks per Department", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    for bar, val in zip(bars, counts.values):
        ax.text(
            val + counts.max() * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    save_fig(fig, "department_distribution", subdir="eda")
    return fig


def timeline_plot(df: pd.DataFrame) -> plt.Figure:
    """Histogram of creation decades stacked/colored by top 5 departments."""
    df2 = df.dropna(subset=["creation_decade"]).copy()
    df2["creation_decade"] = df2["creation_decade"].astype(int)

    top5 = df[TARGET_COL].value_counts().head(5).index
    df2["dept_group"] = df2[TARGET_COL].where(df2[TARGET_COL].isin(top5), "Other")

    palette = dict(zip(list(top5) + ["Other"], sns.color_palette(PLOT_PALETTE, 6)))

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for dept, grp in df2.groupby("dept_group"):
        ax.hist(
            grp["creation_decade"],
            bins=range(1850, 2030, 10),
            alpha=0.6,
            label=dept,
            color=palette.get(dept, "gray"),
            edgecolor="none",
        )

    ax.set_xlabel("Creation Decade")
    ax.set_ylabel("Number of Artworks")
    ax.set_title("Artworks by Creation Decade", fontweight="bold")
    ax.legend(
        title="Department", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8
    )
    fig.tight_layout()
    save_fig(fig, "timeline_by_decade", subdir="eda")
    return fig


def nationality_chart(df: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    """Horizontal bar of top artist nationalities."""
    nat_col = df.get("ArtistNationality", df.get("Nationality")).dropna().str.strip()
    counts = nat_col.value_counts().head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    ax.barh(
        counts.index, counts.values, color=sns.color_palette("Blues_r", len(counts))
    )
    ax.set_xlabel("Number of Artists")
    ax.set_title(f"Top {top_n} Artist Nationalities in MoMA", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    save_fig(fig, "nationality_chart", subdir="eda")
    return fig


def gender_split(df: pd.DataFrame) -> plt.Figure:
    """Donut chart of artist gender breakdown."""
    gender_col = df.get("ArtistGender", df.get("Gender", pd.Series(dtype=str)))
    gender_col = gender_col.fillna("Unknown").str.strip()

    # simplify labels
    gender_col = gender_col.map(
        lambda x: (
            "Female"
            if "female" in x.lower()
            else ("Male" if "male" in x.lower() else "Unknown")
        )
    )
    counts = gender_col.value_counts()

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["#4C72B0", "#DD8452", "#929591"]
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors[: len(counts)],
        wedgeprops=dict(width=0.5),
    )
    for t in autotexts:
        t.set_fontsize(10)
    ax.set_title("Artist Gender Distribution", fontweight="bold", pad=20)
    fig.tight_layout()
    save_fig(fig, "gender_split", subdir="eda")
    return fig


def missing_values_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Heatmap of % missing values per column."""
    miss_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    miss_pct = miss_pct[miss_pct > 0].head(30)

    fig, ax = plt.subplots(figsize=(8, max(4, len(miss_pct) * 0.3)))
    bars = ax.barh(
        miss_pct.index,
        miss_pct.values,
        color=sns.color_palette("Reds_r", len(miss_pct)),
    )
    ax.set_xlabel("% Missing")
    ax.set_title("Missing Value Rate by Column", fontweight="bold")
    ax.set_xlim(0, 105)
    for bar, val in zip(bars, miss_pct.values):
        ax.text(
            val + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    save_fig(fig, "missing_values", subdir="eda")
    return fig


def medium_top20(df: pd.DataFrame) -> plt.Figure:
    """Top-20 most common words/phrases in the Medium column."""
    from collections import Counter

    medium_series = df["Medium"].dropna().astype(str).str.lower()
    words = medium_series.str.split(r"[,;]+", expand=True).stack().str.strip()
    words = words[words.str.len() > 3]
    top = pd.Series(Counter(words)).sort_values(ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    ax.barh(top.index, top.values, color=sns.color_palette("Greens_r", 20))
    ax.set_xlabel("Occurrences")
    ax.set_title("Top 20 Medium Terms", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    save_fig(fig, "medium_top20", subdir="eda")
    return fig


def acquisition_trend(df: pd.DataFrame) -> plt.Figure:
    """Line chart of acquisitions per year — shows MoMA's collecting activity."""
    acq = df.dropna(subset=["acquisition_year"]).copy()
    acq["acquisition_year"] = acq["acquisition_year"].astype(int)
    by_year = acq.groupby("acquisition_year").size()
    by_year = by_year[(by_year.index >= 1920) & (by_year.index <= 2025)]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.fill_between(by_year.index, by_year.values, alpha=0.3, color="#4C72B0")
    ax.plot(by_year.index, by_year.values, color="#4C72B0", linewidth=1.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Artworks Acquired")
    ax.set_title("MoMA Acquisition Activity Over Time", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    save_fig(fig, "acquisition_trend", subdir="eda")
    return fig


def correlation_matrix(feature_df: pd.DataFrame) -> plt.Figure:
    """Heatmap of correlations among the non-TF-IDF numeric features."""
    # Only the 'hand-crafted' columns, not the 100 TF-IDF ones
    non_tfidf = [
        c for c in feature_df.columns if not c.startswith("med_") and c != "target"
    ]
    corr = feature_df[non_tfidf].corr()

    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.3,
        annot_kws={"size": 8},
    )
    ax.set_title("Feature Correlation Matrix", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "correlation_matrix", subdir="eda")
    return fig


def dept_by_decade_heatmap(df: pd.DataFrame) -> plt.Figure:
    """2D heatmap: departments (rows) × decades (cols) = artwork count."""
    df2 = df.dropna(subset=["creation_decade"]).copy()
    df2["creation_decade"] = df2["creation_decade"].astype(int)
    df2 = df2[(df2["creation_decade"] >= 1860) & (df2["creation_decade"] <= 2020)]

    pivot = df2.pivot_table(
        index=TARGET_COL,
        columns="creation_decade",
        values="Title",
        aggfunc="count",
        fill_value=0,
    )

    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(
        pivot, ax=ax, cmap="YlOrRd", linewidths=0.2, cbar_kws={"label": "Artwork Count"}
    )
    ax.set_title("Department × Creation Decade Heatmap", fontweight="bold")
    ax.set_xlabel("Decade")
    ax.set_ylabel("")
    fig.tight_layout()
    save_fig(fig, "dept_by_decade_heatmap", subdir="eda")
    return fig


# ── Run everything ─────────────────────────────────────────────────────────


def all_eda(df: pd.DataFrame, feature_df: pd.DataFrame = None):
    """
    Run every EDA plot. Pass feature_df to also get the correlation matrix.
    Returns a dict of {name: figure}.
    """
    from src.utils import print_section

    print_section("Exploratory Data Analysis")

    figs = {}
    plots = [
        ("department_distribution", lambda: department_distribution(df)),
        ("timeline", lambda: timeline_plot(df)),
        ("nationality", lambda: nationality_chart(df)),
        ("gender", lambda: gender_split(df)),
        ("missing_values", lambda: missing_values_heatmap(df)),
        ("medium_top20", lambda: medium_top20(df)),
        ("acquisition_trend", lambda: acquisition_trend(df)),
        ("dept_by_decade", lambda: dept_by_decade_heatmap(df)),
    ]
    if feature_df is not None:
        plots.append(("correlation", lambda: correlation_matrix(feature_df)))

    import matplotlib.pyplot as plt

    for name, fn in plots:
        print(f"  plotting {name}...")
        fig = fn()
        figs[name] = fig
        plt.close(fig)  # free memory in script mode; notebooks never call all_eda()

    print(f"\n✓ {len(figs)} plots saved to outputs/eda/")
    return figs
