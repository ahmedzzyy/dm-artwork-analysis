"""
app.py — MoMA Art Mining · Streamlit Application

Two modes:
  🎨  Artwork Predictor  — type in artwork details, get a live department prediction
  📊  Project Findings   — guided presentation of all results and insights

Run with:
    uv run streamlit run app.py
"""

import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

from src.config import OUTPUTS_DIR, MODELS_DIR
from src.predictor import TOP_NATIONALITIES

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MoMA Art Mining",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0f0f0f;
    border-right: 1px solid #222;
  }
  [data-testid="stSidebar"] * { color: #e8e8e8 !important; }
  [data-testid="stSidebar"] .stRadio label {
    font-size: 15px;
    padding: 6px 0;
    letter-spacing: 0.01em;
  }

  /* Main bg */
  .stApp { background: #fafaf8; }

  /* Hero title */
  .hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    font-weight: 400;
    color: #0f0f0f;
    line-height: 1.1;
    margin-bottom: 0.2rem;
  }
  .hero-sub {
    font-size: 1rem;
    color: #666;
    font-weight: 300;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 2rem;
  }

  /* Prediction card */
  .pred-card {
    background: #0f0f0f;
    color: #fafaf8;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin: 1rem 0;
  }
  .pred-dept {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #f5c842;
    margin-bottom: 0.3rem;
  }
  .pred-conf {
    font-size: 0.95rem;
    color: #aaa;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }

  /* Finding section headers */
  .section-pill {
    display: inline-block;
    background: #0f0f0f;
    color: #f5c842;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 0.8rem;
  }

  /* Metric cards */
  .metric-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
  .metric-card {
    background: white;
    border: 1px solid #e8e8e0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    flex: 1;
    min-width: 140px;
  }
  .metric-val {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #0f0f0f;
    line-height: 1;
  }
  .metric-label {
    font-size: 0.8rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 4px;
  }

  /* Insight box */
  .insight-box {
    background: #fffbea;
    border-left: 4px solid #f5c842;
    padding: 1rem 1.4rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
    font-size: 0.95rem;
    color: #333;
  }

  /* Divider */
  .divider { border: none; border-top: 1px solid #e8e8e0; margin: 2rem 0; }

  /* Hide streamlit branding */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def load_image(name: str, subdir: str = "figures") -> Image.Image | None:
    path = OUTPUTS_DIR / subdir / f"{name}.png"
    if path.exists():
        return Image.open(path)
    return None


def img_or_warn(name: str, subdir: str = "figures", caption: str = ""):
    img = load_image(name, subdir)
    if img:
        st.image(img, caption=caption, use_container_width=True)
    else:
        st.info(f"📂 Run `main.py` first to generate **{name}.png**")


def insight(text: str):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


def pill(text: str):
    st.markdown(f'<div class="section-pill">{text}</div>', unsafe_allow_html=True)


def metric_cards(items: list[tuple[str, str]]):
    """items = [(value, label), ...]"""
    cols = st.columns(len(items))
    for col, (val, label) in zip(cols, items):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-val">{val}</div>'
            f'<div class="metric-label">{label}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<p style=\"font-family:'DM Serif Display',serif;font-size:1.4rem;"
        'color:#f5c842;margin-bottom:0">MoMA</p>'
        '<p style="font-size:0.75rem;color:#888;letter-spacing:0.1em;'
        'text-transform:uppercase;margin-top:0">Art Mining</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    mode = st.radio(
        "Navigate",
        ["🎨  Artwork Predictor", "📊  Project Findings"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.75rem;color:#555;line-height:1.6">'
        "Museum of Modern Art · 160K artworks<br>"
        "Random Forest · 94.8% accuracy<br>"
        "KMeans + DBSCAN clustering"
        "</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MODE 1 — ARTWORK PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

if mode == "🎨  Artwork Predictor":
    st.markdown(
        '<h1 class="hero-title">Artwork Department<br>Predictor</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-sub">Describe an artwork · Get an instant classification</p>',
        unsafe_allow_html=True,
    )

    # Check models exist before trying to load
    models_ready = (
        bool(list(MODELS_DIR.glob("best_classifier_*.pkl")))
        and (MODELS_DIR / "feature_scaler.pkl").exists()
        and (MODELS_DIR / "feature_meta.pkl").exists()
    )

    if not models_ready:
        st.error(
            "**Models not found.** Run the training pipeline first:\n\n"
            "```\nuv run python main.py\n```\n\n"
            "This trains all models and saves them to `outputs/models/`."
        )
        st.stop()

    # Load predictor (cached after first load)
    @st.cache_resource
    def load_predictor():
        from src.predictor import get_predictor

        return get_predictor()

    predictor = load_predictor()

    # ── Input form ──────────────────────────────────────────────────────────
    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("#### Artwork Details")

        medium = st.text_input(
            "Medium",
            value="Oil on canvas",
            placeholder="e.g. Gelatin silver print, Lithograph, Oil on canvas",
            help="The material / technique of the artwork. This is the strongest predictor.",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            creation_year = st.number_input(
                "Creation Year", min_value=1800, max_value=2024, value=1965
            )
        with col_b:
            acquisition_year = st.number_input(
                "Acquisition Year", min_value=1800, max_value=2024, value=1975
            )

        nationality = st.selectbox("Artist Nationality", TOP_NATIONALITIES, index=0)
        is_female = st.checkbox("Artist is female")

        st.markdown("#### Dimensions *(optional)*")
        col_h, col_w = st.columns(2)
        with col_h:
            height_cm = st.number_input(
                "Height (cm)", min_value=0.0, value=0.0, step=1.0
            )
        with col_w:
            width_cm = st.number_input("Width (cm)", min_value=0.0, value=0.0, step=1.0)

        predict_btn = st.button(
            "Predict Department →", type="primary", use_container_width=True
        )

    # ── Result panel ────────────────────────────────────────────────────────
    with col_result:
        if predict_btn or True:  # show placeholder until first click
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

            if "last_result" in st.session_state:
                result = st.session_state["last_result"]
                dept = result["predicted_class"]
                conf = result["confidence"]

                # Prediction card
                st.markdown(
                    f'<div class="pred-card">'
                    f'<div class="pred-conf">Predicted Department</div>'
                    f'<div class="pred-dept">{dept}</div>'
                    f'<div class="pred-conf">{conf:.1%} confidence</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Probability bar chart
                st.markdown("#### All Department Probabilities")
                probs = result["probabilities"]
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                labels = [p[0] for p in sorted_probs]
                values = [p[1] for p in sorted_probs]
                colors = ["#f5c842" if l == dept else "#e0e0d8" for l in labels]

                fig = go.Figure(
                    go.Bar(
                        x=values,
                        y=labels,
                        orientation="h",
                        marker_color=colors,
                        text=[f"{v:.1%}" for v in values],
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    height=300,
                    margin=dict(l=0, r=60, t=10, b=10),
                    xaxis=dict(range=[0, 1], showticklabels=False, showgrid=False),
                    yaxis=dict(showgrid=False),
                    plot_bgcolor="#fafaf8",
                    paper_bgcolor="#fafaf8",
                    font=dict(family="DM Sans", size=13),
                    showlegend=False,
                )
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)

                # Top features driving this prediction
                st.markdown("#### What drove this prediction")
                top_features = result["top_features"][:8]
                feat_names = [f[0].replace("med_", "medium: ") for f in top_features]
                feat_vals = [f[1] for f in top_features]

                fig2 = go.Figure(
                    go.Bar(
                        x=feat_vals,
                        y=feat_names,
                        orientation="h",
                        marker_color="#0f0f0f",
                    )
                )
                fig2.update_layout(
                    height=260,
                    margin=dict(l=0, r=20, t=10, b=10),
                    xaxis=dict(showticklabels=False, showgrid=False),
                    yaxis=dict(showgrid=False),
                    plot_bgcolor="#fafaf8",
                    paper_bgcolor="#fafaf8",
                    font=dict(family="DM Sans", size=12),
                )
                fig2.update_yaxes(autorange="reversed")
                st.plotly_chart(fig2, use_container_width=True)

            else:
                st.markdown(
                    '<div style="height:300px;display:flex;align-items:center;'
                    "justify-content:center;color:#aaa;font-size:1rem;"
                    'border:1px dashed #ddd;border-radius:12px">'
                    "Fill in the form and click Predict →"
                    "</div>",
                    unsafe_allow_html=True,
                )

    # ── Example presets ─────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("#### Try these examples")
    ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)

    examples = [
        ("📸 Photograph", "Gelatin silver print", "American", 1975, 1980),
        ("🖼️ Painting", "Oil on canvas", "French", 1923, 1940),
        ("📐 Architecture", "Graphite on paper", "Swiss", 1968, 1972),
        ("🎬 Film", "16mm film, color, sound", "Japanese", 1982, 1990),
    ]
    for col, (label, med, nat, cy, ay) in zip(
        [ex_col1, ex_col2, ex_col3, ex_col4], examples
    ):
        if col.button(label, use_container_width=True):
            with st.spinner("Classifying..."):
                from src.predictor import get_predictor

                p = get_predictor()
                r = p.predict(
                    medium=med, nationality=nat, creation_year=cy, acquisition_year=ay
                )
            st.session_state["last_result"] = r
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MODE 2 — PROJECT FINDINGS
# ══════════════════════════════════════════════════════════════════════════════

else:
    st.markdown('<h1 class="hero-title">Project Findings</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Discovering artistic movements through machine learning</p>',
        unsafe_allow_html=True,
    )

    # ── Summary metrics ──────────────────────────────────────────────────────
    metric_cards(
        [
            ("160K", "Artworks analysed"),
            ("7", "Departments classified"),
            ("94.8%", "Best model accuracy"),
            ("110", "Features engineered"),
            ("3", "ML techniques applied"),
        ]
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📁 Dataset", "🔍 EDA", "🤖 Classification", "🔵 Clustering", "📌 Conclusions"]
    )

    # ── Tab 1: Dataset ────────────────────────────────────────────────────────
    with tab1:
        pill("Dataset Overview")
        st.markdown("### Museum of Modern Art — MoMA Collection")
        st.markdown(
            "The dataset is publicly released by MoMA for research. "
            "It contains every artwork in their permanent collection with rich metadata."
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Artworks CSV**")
            st.markdown("""
- 160,269 artworks
- Title, Medium, Dimensions, Date, Department
- Date Acquired, URL, ConstituentID
            """)
        with col2:
            st.markdown("**Artists CSV**")
            st.markdown("""
- 15,807 artists
- Name, Nationality, Gender
- Birth year, Death year
- Linked via ConstituentID
            """)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        pill("After Cleaning")
        col3, col4, col5 = st.columns(3)
        col3.metric("Rows retained", "160,238", "99.98% kept")
        col4.metric("Departments", "7", "after min-sample filter")
        col5.metric("Features built", "110", "10 structured + 100 TF-IDF")

        insight(
            "The Medium column — e.g. 'Gelatin silver print', 'Oil on canvas' — "
            "turned out to be the most information-dense field in the entire dataset. "
            "TF-IDF on this column alone accounts for ~90 of our 110 features."
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        pill("Class Distribution")
        img_or_warn("department_distribution", subdir="eda")
        insight(
            "The dataset is heavily imbalanced — Drawings & Prints alone is 51% of all artworks. "
            "We handled this with stratified train/test splitting to ensure every class "
            "is proportionally represented in both sets."
        )

    # ── Tab 2: EDA ────────────────────────────────────────────────────────────
    with tab2:
        pill("Exploratory Data Analysis")
        st.markdown("### What the data tells us before any modelling")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Creation decade timeline**")
            img_or_warn("timeline_by_decade", subdir="eda")
            insight(
                "Most artworks were created post-1950, reflecting MoMA's modern/contemporary focus."
            )
        with col2:
            st.markdown("**Acquisition activity over time**")
            img_or_warn("acquisition_trend", subdir="eda")
            insight(
                "Acquisition spikes in the 1960s and 2000s correspond to major collection drives."
            )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Artist nationalities**")
            img_or_warn("nationality_chart", subdir="eda")
        with col4:
            st.markdown("**Artist gender distribution**")
            img_or_warn("gender_split", subdir="eda")

        insight(
            "~85% of represented artists are male. This demographic skew is a known "
            "bias in major Western museum collections — not a data quality issue, "
            "but a cultural one. Our model will reflect this distribution."
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("**Department × Creation Decade heatmap**")
        img_or_warn("dept_by_decade_heatmap", subdir="eda")
        insight(
            "Photography barely exists before 1950 then explodes — confirming that "
            "creation decade is a genuinely useful feature for classification. "
            "Different departments have distinct temporal signatures."
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("**Top 20 medium terms**")
        img_or_warn("medium_top20", subdir="eda")
        insight(
            "The most common medium terms — 'print', 'paper', 'gelatin silver' — "
            "map almost directly to specific departments. This is why TF-IDF on the "
            "Medium column is so powerful for classification."
        )

    # ── Tab 3: Classification ─────────────────────────────────────────────────
    with tab3:
        pill("Supervised Learning")
        st.markdown("### Three models, one winner")
        st.markdown(
            "We trained three classifiers on the same 110-feature matrix "
            "with an 80/20 stratified train/test split."
        )

        # Model comparison table
        model_data = pd.DataFrame(
            {
                "Model": ["Logistic Regression", "SVM (RBF)", "Random Forest"],
                "Accuracy": ["86.0%", "88.3%", "94.8%"],
                "Macro F1": ["0.68", "0.60", "0.86"],
                "Training time": ["26s", "56s", "6s"],
                "Notes": [
                    "Linear baseline — fast, interpretable",
                    "Strong on TF-IDF but ignores minority classes entirely",
                    "Best overall — fast to train, handles imbalance well",
                ],
            }
        )
        st.dataframe(model_data, hide_index=True, use_container_width=True)

        insight(
            "Random Forest wins on every metric and trains the fastest of the three. "
            "SVM scored 0.00 F1 on Film and Fluxus — it simply ignored minority classes "
            "because its 20K subsample didn't see enough of them."
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Model accuracy comparison**")
            img_or_warn("model_comparison", subdir="figures")
        with col2:
            st.markdown("**Feature importances — Random Forest**")
            img_or_warn("feature_importances", subdir="figures")

        insight(
            "The top features are almost entirely TF-IDF medium terms — "
            "'gelatin silver', 'print', 'offset'. The model essentially learned that "
            "artistic medium is the primary determinant of department, "
            "which validates the museum's own cataloguing logic."
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("**Confusion matrices**")
        cf_col1, cf_col2, cf_col3 = st.columns(3)
        with cf_col1:
            st.markdown("*Logistic Regression*")
            img_or_warn("confusion_logistic_regression", subdir="figures")
        with cf_col2:
            st.markdown("*Random Forest*")
            img_or_warn("confusion_random_forest", subdir="figures")
        with cf_col3:
            st.markdown("*SVM*")
            img_or_warn("confusion_svm", subdir="figures")

        insight(
            "The RF confusion matrix shows most mistakes happen at the "
            "Architecture & Design ↔ Drawings & Prints boundary — "
            "both use paper-based media, so the model genuinely struggles "
            "where human curators would too."
        )

    # ── Tab 4: Clustering ─────────────────────────────────────────────────────
    with tab4:
        pill("Unsupervised Learning")
        st.markdown("### What groupings does the data suggest — without being told?")
        st.markdown(
            "We removed the department labels entirely and let two algorithms "
            "find structure on their own. The question: do natural groupings "
            "align with the museum's official departments?"
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        pill("Dimensionality Reduction")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**PCA — choosing how many components**")
            img_or_warn("pca_variance", subdir="figures")
            insight(
                "50 PCA components capture 84.5% of variance — compact enough for fast clustering."
            )
        with col2:
            st.markdown("**Elbow + Silhouette — choosing k**")
            img_or_warn("elbow_silhouette", subdir="figures")
            insight(
                "We enforce k ≥ 3 to avoid trivial two-way splits. The best silhouette above that threshold determines k."
            )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        pill("KMeans Results")
        col3, col4 = st.columns(2)
        with col3:
            img_or_warn("pca_—_kmeans_k=6", subdir="figures")
        with col4:
            img_or_warn("cluster_dept_breakdown", subdir="figures")

        insight(
            "KMeans clusters partially but not perfectly align with departments. "
            "Some clusters map cleanly to Photography or Architecture; "
            "others are cross-departmental — grouping by era rather than medium. "
            "That cross-departmental clustering is the interesting finding."
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        pill("DBSCAN Results")
        col5, col6 = st.columns(2)
        with col5:
            img_or_warn("pca_—_dbscan", subdir="figures")
        with col6:
            st.markdown("**KMeans vs DBSCAN**")
            comparison = pd.DataFrame(
                {
                    "Property": [
                        "Assigns every point",
                        "Finds noise",
                        "Shape assumption",
                        "Need to set k",
                    ],
                    "KMeans": ["✅ Yes", "❌ No", "Spherical", "✅ Yes"],
                    "DBSCAN": ["❌ No", "✅ Yes", "Arbitrary", "❌ No"],
                }
            )
            st.dataframe(comparison, hide_index=True, use_container_width=True)
            insight(
                "DBSCAN labels ~X% of artworks as noise — "
                "these are genuinely unusual, cross-disciplinary works that don't "
                "fit any cluster. KMeans would have forcibly assigned them somewhere."
            )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        pill("t-SNE Visualisation")
        img_or_warn("t-sne_—_kmeans_k=6", subdir="figures")
        insight(
            "t-SNE preserves local neighbourhood structure — points close together here "
            "were genuinely similar in the original 110-dimensional space. "
            "The visible island structures confirm the clusters are real, not arbitrary."
        )

    # ── Tab 5: Conclusions ────────────────────────────────────────────────────
    with tab5:
        pill("Conclusions")
        st.markdown("### What did we actually find?")

        st.markdown("#### Finding 1 — Medium predicts department almost perfectly")
        st.markdown(
            "The top feature importances are all medium terms. "
            "'Gelatin silver' → Photography with 96% confidence. "
            "'Oil on canvas' → Painting & Sculpture with 89% confidence. "
            "The museum's departmental system is essentially a medium-based taxonomy — "
            "our model discovered this automatically."
        )

        st.markdown("#### Finding 2 — The hard cases are genuinely ambiguous")
        st.markdown(
            "Architecture & Design and Drawings & Prints share media (graphite, paper, ink). "
            "The model's mistakes cluster exactly here — and a human curator "
            "looking at the same artwork would sometimes disagree too. "
            "The confusion matrix maps the boundaries of human curatorial logic."
        )

        st.markdown(
            "#### Finding 3 — Unsupervised clusters cut across departments by era"
        )
        st.markdown(
            "KMeans found groups that don't follow department lines. "
            "One cluster gathers 1960s–70s works across Photography, Drawing, and Architecture — "
            "likely the Conceptual Art period, which resisted medium-based categorisation. "
            "This is something the classification model can't surface."
        )

        st.markdown("#### Finding 4 — The collection encodes historical biases")
        st.markdown(
            "85% male artists, heavy American/European skew. "
            "Any classifier trained on this data will perpetuate these patterns. "
            "A production deployment would need bias auditing — "
            "accuracy on the test set does not mean fairness across demographics."
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        pill("Summary")
        metric_cards(
            [
                ("94.8%", "RF Accuracy"),
                ("0.86", "Macro F1"),
                ("~96%", "Medium→Dept confidence"),
                ("3", "DM techniques"),
            ]
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#888;font-size:0.85rem">'
            "Ahmed Sahigara · 230911180 · School of Computer Engineering · MIT Manipal"
            "</p>",
            unsafe_allow_html=True,
        )
