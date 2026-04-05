"""
app.py — MoMA Art Mining · Streamlit Application

Two modes:
  🎨  Artwork Predictor  — live department prediction from artwork details
  📊  Project Findings   — guided presentation of all results

Run:
    uv run streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import MODELS_DIR, OUTPUTS_DIR
from src.predictor import TOP_NATIONALITIES

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MoMA Art Mining",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal CSS — only things Streamlit's theme system genuinely can't do:
# 1. Hide the default top bar chrome
# 2. Slightly round the prediction result card
st.markdown(
    """
<style>
  #MainMenu, footer, header { visibility: hidden; }
  .pred-card {
    background-color: #1a1a1a;
    color: #fafaf8;
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin-bottom: 1rem;
  }
  .pred-dept  { font-size: 1.9rem; font-weight: 600; color: #c8a951; margin-bottom: 0.2rem; }
  .pred-label { font-size: 0.8rem; color: #aaa; letter-spacing: 0.08em; text-transform: uppercase; }
  .insight {
    border-left: 3px solid #c8a951;
    padding: 0.7rem 1rem;
    margin: 1rem 0;
    background: #f7f4e8;
    border-radius: 0 8px 8px 0;
    font-size: 0.92rem;
    color: #333;
  }
</style>
""",
    unsafe_allow_html=True,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def load_img(name: str, subdir: str = "figures"):
    from PIL import Image

    p = OUTPUTS_DIR / subdir / f"{name}.png"
    return Image.open(p) if p.exists() else None


def show_img(name: str, subdir: str = "figures", caption: str = ""):
    img = load_img(name, subdir)
    if img:
        st.image(img, caption=caption, use_container_width=True)
    else:
        st.info(f"Run `main.py` first to generate **{name}.png**", icon="📂")


def insight(text: str):
    st.markdown(f'<div class="insight">💡 {text}</div>', unsafe_allow_html=True)


def metric_row(items: list[tuple[str, str, str]]):
    """items = [(value, label, delta_str), ...]  delta_str can be empty"""
    for col, (val, label, delta) in zip(st.columns(len(items)), items):
        col.metric(label, val, delta or None)


# ── Check models ───────────────────────────────────────────────────────────


def models_ready() -> bool:
    return (
        bool(list(MODELS_DIR.glob("best_classifier_*.pkl")))
        and (MODELS_DIR / "feature_scaler.pkl").exists()
        and (MODELS_DIR / "feature_meta.pkl").exists()
    )


# ── Load predictor once (Streamlit caches across reruns) ──────────────────


@st.cache_resource
def load_predictor():
    from src.predictor import get_predictor

    return get_predictor()


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("MoMA Art Mining")
    st.caption("MIT Manipal · Data Mining Project")
    st.divider()
    mode = st.radio(
        "Mode",
        ["🎨  Artwork Predictor", "📊  Project Findings"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("160,238 artworks · 7 departments · Random Forest 94.8%")


# ══════════════════════════════════════════════════════════════════════════
#  MODE 1 — PREDICTOR
# ══════════════════════════════════════════════════════════════════════════

if mode == "🎨  Artwork Predictor":
    st.title("Artwork Department Predictor")
    st.caption(
        "Describe an artwork and the model predicts which MoMA department it belongs to."
    )
    st.divider()

    if not models_ready():
        st.error(
            "**Models not found.** Train first:\n```\nuv run python main.py\n```",
            icon="🚨",
        )
        st.stop()

    predictor = load_predictor()

    # ── Layout: form left, results right ──────────────────────────────────
    form_col, result_col = st.columns([1, 1], gap="large")

    with form_col:
        st.subheader("Artwork Details")

        medium = st.text_input(
            "Medium *",
            value="Oil on canvas",
            help="The material / technique — this is the strongest predictor by far.",
        )

        c1, c2 = st.columns(2)
        creation_year = c1.number_input("Creation year", 1800, 2024, 1965)
        acquisition_year = c2.number_input("Acquisition year", 1800, 2024, 1975)

        nationality = st.selectbox("Artist nationality", TOP_NATIONALITIES)
        is_female = st.checkbox("Artist is female")

        with st.expander("Dimensions (optional)"):
            dc1, dc2 = st.columns(2)
            height_cm = dc1.number_input("Height (cm)", 0.0, 10000.0, 0.0)
            width_cm = dc2.number_input("Width (cm)", 0.0, 10000.0, 0.0)

        predict_btn = st.button("Predict →", type="primary", use_container_width=True)

    # ── Quick-pick examples ────────────────────────────────────────────────
    with form_col:
        st.caption("Quick examples:")
        ex_cols = st.columns(4)
        examples = [
            ("📸 Photo", "Gelatin silver print", "American", 1975, 1980),
            ("🖼️ Paint", "Oil on canvas", "French", 1923, 1940),
            ("📐 Arch", "Graphite on paper", "Swiss", 1968, 1972),
            ("🎬 Film", "16mm film, color, sound", "Japanese", 1982, 1990),
        ]
        for col, (label, med, nat, cy, ay) in zip(ex_cols, examples):
            if col.button(label, use_container_width=True):
                with st.spinner("Classifying..."):
                    r = predictor.predict(
                        medium=med,
                        nationality=nat,
                        creation_year=cy,
                        acquisition_year=ay,
                    )
                st.session_state["last_result"] = r
                st.rerun()

    # ── Run prediction ─────────────────────────────────────────────────────
    if predict_btn:
        with st.spinner("Classifying..."):
            result = predictor.predict(
                medium=medium,
                nationality=nationality,
                creation_year=int(creation_year),
                acquisition_year=int(acquisition_year),
                is_female=is_female,
                height_cm=float(height_cm),
                width_cm=float(width_cm),
            )
        st.session_state["last_result"] = result

    # ── Result display ─────────────────────────────────────────────────────
    with result_col:
        if "last_result" not in st.session_state:
            st.info("Fill in the form and click **Predict →**", icon="👈")
        else:
            res = st.session_state["last_result"]
            dept = res["predicted_class"]
            conf = res["confidence"]

            st.markdown(
                f'<div class="pred-card">'
                f'<div class="pred-label">Predicted Department</div>'
                f'<div class="pred-dept">{dept}</div>'
                f'<div class="pred-label">Confidence &nbsp;·&nbsp; {conf:.1%}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

            # All-departments probability chart
            st.caption("Probability across all departments")
            probs = dict(sorted(res["probabilities"].items(), key=lambda x: x[1]))
            colors = ["#c8a951" if k == dept else "#d0cfc8" for k in probs]
            fig = go.Figure(
                go.Bar(
                    x=list(probs.values()),
                    y=list(probs.keys()),
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:.1%}" for v in probs.values()],
                    textposition="outside",
                )
            )
            fig.update_layout(
                height=260,
                margin=dict(l=0, r=55, t=4, b=4),
                xaxis=dict(
                    range=[0, 1.05],
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                ),
                yaxis=dict(showgrid=False),
                plot_bgcolor="#fafaf8",
                paper_bgcolor="#fafaf8",
                font=dict(size=12),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Top driving features
            st.caption("Features that drove this prediction")
            feats = res["top_features"][:8]
            f_names = [f[0].replace("med_", "medium: ") for f in feats]
            f_vals = [f[1] for f in feats]
            fig2 = go.Figure(
                go.Bar(
                    x=f_vals,
                    y=f_names,
                    orientation="h",
                    marker_color="#1a1a1a",
                )
            )
            fig2.update_layout(
                height=240,
                margin=dict(l=0, r=10, t=4, b=4),
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False),
                plot_bgcolor="#fafaf8",
                paper_bgcolor="#fafaf8",
                font=dict(size=12),
            )
            fig2.update_yaxes(autorange="reversed")
            st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
#  MODE 2 — FINDINGS
# ══════════════════════════════════════════════════════════════════════════

else:
    st.title("Project Findings")
    st.caption(
        "Discovering artistic movements through machine learning · MoMA Collection"
    )
    st.divider()

    metric_row(
        [
            ("160,238", "Artworks analysed", ""),
            ("7", "Departments", ""),
            ("94.8%", "Best accuracy", "Random Forest"),
            ("110", "Features", "10 structured + 100 TF-IDF"),
            ("3", "DM techniques", "Classification · Clustering · Dim. reduction"),
        ]
    )

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📁 Dataset", "🔍 EDA", "🤖 Classification", "🔵 Clustering", "📌 Conclusions"]
    )

    # ── Dataset ────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Museum of Modern Art — Collection Dataset")
        st.markdown(
            "Publicly released by MoMA for research. "
            "Two CSVs merged on `ConstituentID`."
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Artworks.csv** — 160,269 rows")
            st.markdown(
                "Title · Medium · Dimensions · Date · Department · DateAcquired"
            )
        with c2:
            st.markdown("**Artists.csv** — 15,807 rows")
            st.markdown("Name · Nationality · Gender · Birth year · Death year")

        st.divider()
        st.subheader("After cleaning")
        metric_row(
            [
                ("160,238", "Rows retained", "99.98% kept"),
                ("7", "Departments", "after min-sample filter"),
                ("110", "Features", "10 structured + 100 TF-IDF"),
            ]
        )
        insight(
            "The Medium column — 'Gelatin silver print', 'Oil on canvas' — "
            "is the most information-dense field. TF-IDF on this single column "
            "accounts for 90 of our 110 features and drives most of the model's accuracy."
        )
        st.divider()
        st.subheader("Class distribution")
        show_img("department_distribution", subdir="eda")
        insight(
            "Drawings & Prints is 51% of all artworks — heavy class imbalance. "
            "Handled with stratified splitting so every class is proportionally "
            "represented in both train and test sets."
        )

    # ── EDA ────────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Exploratory Data Analysis")
        st.markdown("Understanding the data before any modelling.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Creation decade timeline**")
            show_img("timeline_by_decade", subdir="eda")
            insight(
                "Most artworks post-1950 — reflects MoMA's modern/contemporary mandate."
            )
        with c2:
            st.markdown("**Acquisition activity**")
            show_img("acquisition_trend", subdir="eda")
            insight(
                "Spikes in the 1960s and 2000s correspond to major collection drives."
            )

        st.divider()
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Artist nationalities (top 20)**")
            show_img("nationality_chart", subdir="eda")
        with c4:
            st.markdown("**Artist gender split**")
            show_img("gender_split", subdir="eda")

        insight(
            "~85% of represented artists are male — a known bias in major Western collections. "
            "Our model reflects this distribution; it is not a data quality problem, it is a cultural one."
        )

        st.divider()
        st.markdown("**Department × Creation Decade heatmap**")
        show_img("dept_by_decade_heatmap", subdir="eda")
        insight(
            "Photography barely exists before 1950 then explodes. "
            "Each department has a distinct temporal signature — "
            "confirming that creation decade will be a genuinely useful feature."
        )

        st.divider()
        st.markdown("**Top 20 medium terms**")
        show_img("medium_top20", subdir="eda")
        insight(
            "'print', 'paper', 'gelatin silver' map almost directly to specific departments. "
            "This is the intuition behind using TF-IDF on the Medium column."
        )

    # ── Classification ─────────────────────────────────────────────────────
    with tab3:
        st.subheader("Classification — Supervised Learning")
        st.markdown(
            "Three classifiers trained on the same 110-feature matrix. "
            "80 / 20 stratified train-test split."
        )

        results_df = pd.DataFrame(
            {
                "Model": ["Logistic Regression", "SVM (RBF kernel)", "Random Forest"],
                "Test Accuracy": ["86.0%", "88.3%", "**94.8%**"],
                "Macro F1": ["0.68", "0.60", "**0.86**"],
                "Train time": ["26s", "56s", "6s"],
                "Notes": [
                    "Linear baseline — fast and interpretable",
                    "Ignored minority classes entirely (Film, Fluxus → 0% recall)",
                    "Best on every metric — and fastest to train",
                ],
            }
        )
        st.dataframe(results_df, hide_index=True, use_container_width=True)

        insight(
            "SVM scored 0.00 F1 on Film and Fluxus Collection. "
            "It was subsampled to 20K for speed — and those classes are so rare "
            "they barely appeared in the sample. Random Forest trained on the full set "
            "had no such problem."
        )

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Model accuracy comparison**")
            show_img("model_comparison")
        with c2:
            st.markdown("**Top feature importances — Random Forest**")
            show_img("feature_importances")

        insight(
            "The top features are almost entirely TF-IDF medium terms. "
            "The model learned that artistic medium is the primary determinant of department — "
            "it discovered the museum's own cataloguing logic automatically."
        )

        st.divider()
        st.markdown("**Confusion matrices**")
        cf1, cf2, cf3 = st.columns(3)
        with cf1:
            st.caption("Logistic Regression")
            show_img("confusion_logistic_regression")
        with cf2:
            st.caption("Random Forest")
            show_img("confusion_random_forest")
        with cf3:
            st.caption("SVM")
            show_img("confusion_svm")

        insight(
            "Most RF mistakes happen at the Architecture & Design ↔ Drawings & Prints boundary. "
            "Both use paper-based media. The model is uncertain exactly where human curators are uncertain too."
        )

    # ── Clustering ─────────────────────────────────────────────────────────
    with tab4:
        st.subheader("Clustering — Unsupervised Learning")
        st.markdown(
            "Department labels removed entirely. "
            "Two algorithms find structure on their own — "
            "do natural groupings match MoMA's departments?"
        )

        st.divider()
        st.markdown("**Dimensionality reduction first**")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("PCA variance — choosing n_components")
            show_img("pca_variance")
            insight(
                "50 components capture 84.5% of variance — compact enough for fast clustering."
            )
        with c2:
            st.caption("Elbow + Silhouette — choosing k")
            show_img("elbow_silhouette")
            insight("k ≥ 3 enforced to avoid trivial two-way splits.")

        st.divider()
        st.markdown("**KMeans results**")
        c3, c4 = st.columns(2)
        with c3:
            show_img("pca___kmeans_k_12")
        with c4:
            show_img("cluster_dept_breakdown")

        insight(
            "KMeans clusters partially align with departments but not perfectly. "
            "Some clusters map cleanly to Photography or Architecture. "
            "Others are cross-departmental — grouping by era rather than medium. "
            "That is the interesting finding unsupervised learning surfaces."
        )

        st.divider()
        st.markdown("**DBSCAN results**")
        c5, c6 = st.columns(2)
        with c5:
            show_img("pca___dbscan")
        with c6:
            st.markdown("**KMeans vs DBSCAN**")
            cmp = pd.DataFrame(
                {
                    "Property": [
                        "Assigns every point",
                        "Detects noise/outliers",
                        "Assumes cluster shape",
                        "Requires k upfront",
                    ],
                    "KMeans": ["✅ Yes", "❌ No", "Spherical", "✅ Yes"],
                    "DBSCAN": ["❌ No", "✅ Yes", "Arbitrary", "❌ No"],
                }
            )
            st.dataframe(cmp, hide_index=True, use_container_width=True)
            insight(
                "DBSCAN labels outlier artworks as noise rather than forcing them into a cluster. "
                "These are genuinely unusual, cross-disciplinary works — "
                "KMeans would have distorted a cluster to accommodate them."
            )

        st.divider()
        st.markdown("**t-SNE visualisation**")
        show_img("t_sne___kmeans_k_12")
        insight(
            "t-SNE preserves local neighbourhood structure — points close together here "
            "were genuinely similar in 110-dimensional space. "
            "The visible island structures confirm the clusters are real, not arbitrary."
        )

    # ── Conclusions ────────────────────────────────────────────────────────
    with tab5:
        st.subheader("Conclusions")

        st.markdown("#### Finding 1 — Medium predicts department almost perfectly")
        st.markdown(
            "Top feature importances are all medium terms. "
            "'Gelatin silver' → Photography at 96% confidence. "
            "The museum's departmental system is essentially a medium-based taxonomy. "
            "Our model discovered this automatically — without being told."
        )

        st.divider()
        st.markdown("#### Finding 2 — The hard cases are genuinely ambiguous")
        st.markdown(
            "Architecture & Design and Drawings & Prints share paper-based media. "
            "The model's mistakes cluster exactly here. "
            "A human curator looking at the same edge-case artwork would sometimes disagree too. "
            "The confusion matrix maps the limits of the cataloguing system itself."
        )

        st.divider()
        st.markdown(
            "#### Finding 3 — Unsupervised clusters cross department lines by era"
        )
        st.markdown(
            "KMeans found groups that don't follow department boundaries. "
            "One cluster gathers 1960s–70s works across Photography, Drawing, and Architecture — "
            "likely the Conceptual Art period, which deliberately resisted medium-based categorisation. "
            "Classification cannot surface this. Clustering can."
        )

        st.divider()
        st.markdown("#### Finding 4 — The collection encodes historical biases")
        st.markdown(
            "85% male artists, heavy American and European skew. "
            "A classifier trained on this data will perpetuate these patterns. "
            "Accuracy on a test set drawn from the same distribution does not mean fairness."
        )

        st.divider()
        st.subheader("Summary")
        metric_row(
            [
                ("94.8%", "RF Test Accuracy", ""),
                ("0.86", "Macro F1", "Random Forest"),
                ("~96%", "Medium→Dept confidence", "from feature importances"),
                ("3", "DM techniques", "Classification · Clustering · PCA"),
            ]
        )
        st.divider()
        st.caption(
            "Ahmed Sahigara · 230911180 · School of Computer Engineering · MIT Manipal"
        )
