"""
features.py — turn the cleaned DataFrame into numeric feature matrices
ready for sklearn models.

Two outputs:
  clf_df   — for classification  (includes encoded target)
  clust_df — for clustering       (no target, same features)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import TARGET_COL, MEDIUM_MAX_FEATURES
from src.utils import timer


# ── Individual feature builders ───────────────────────────────────────────


def _date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric features derived from dates."""
    feats = pd.DataFrame(index=df.index)
    feats["creation_year"] = df.get("creation_year", np.nan)
    feats["creation_decade"] = df.get("creation_decade", np.nan)
    feats["acquisition_year"] = df.get("acquisition_year", np.nan)
    feats["years_to_acquire"] = df.get("years_to_acquire", np.nan).clip(-5, 200)

    # Impute medians
    for col in feats.columns:
        feats[col] = feats[col].fillna(feats[col].median())
    return feats


def _artist_features(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric features from artist metadata."""
    feats = pd.DataFrame(index=df.index)
    feats["is_female"] = df.get("is_female", 0).fillna(0).astype(float)
    feats["artist_career_len"] = df.get("artist_career_len", np.nan).clip(0, 100)
    feats["artist_career_len"] = feats["artist_career_len"].fillna(
        feats["artist_career_len"].median()
    )

    # Nationality → frequency encoding (top 20 + "Other")
    nat_col = df.get(
        "ArtistNationality", pd.Series(["Unknown"] * len(df), index=df.index)
    )
    nat_col = nat_col.fillna("Unknown")
    freq = nat_col.value_counts(normalize=True)
    top20 = set(freq.head(20).index)
    feats["nationality_enc"] = nat_col.map(
        lambda x: freq.get(x, 0) if x in top20 else freq.get("Other", 0)
    )

    return feats


def _dimension_features(df: pd.DataFrame) -> pd.DataFrame:
    """Height, width, area — imputed with medians."""
    feats = pd.DataFrame(index=df.index)
    feats["height_cm"] = df.get("height_cm", np.nan)
    feats["width_cm"] = df.get("width_cm", np.nan)
    feats["area_cm2"] = df.get("area_cm2", np.nan)

    for col in feats.columns:
        median = feats[col].median()
        feats[col] = feats[col].fillna(median if pd.notna(median) else 0)

    # Cap extreme outliers at 99th percentile
    for col in feats.columns:
        cap = feats[col].quantile(0.99)
        feats[col] = feats[col].clip(upper=cap)

    return feats


def _medium_features(
    df: pd.DataFrame, max_features: int = MEDIUM_MAX_FEATURES
) -> tuple[pd.DataFrame, TfidfVectorizer]:
    """
    TF-IDF on the Medium column (e.g. 'Oil on canvas', 'Gelatin silver print').
    Returns a sparse-turned-dense DataFrame and the fitted vectorizer.
    """
    medium_series = df.get("Medium", pd.Series([""] * len(df), index=df.index))
    medium_series = medium_series.fillna("").astype(str).str.lower()

    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=5,
        strip_accents="unicode",
    )
    X = vec.fit_transform(medium_series)
    col_names = [f"med_{t}" for t in vec.get_feature_names_out()]
    feats = pd.DataFrame(X.toarray(), columns=col_names, index=df.index)
    return feats, vec


# ── Main feature extractor ────────────────────────────────────────────────


@timer
def extract_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Build full feature matrices.

    Returns
    -------
    clf_df   : DataFrame with features + encoded 'target' column
    clust_df : DataFrame with features only (no target)
    meta     : dict with encoders/vectorizers for inverse transforms
    """
    print("Engineering features...")

    date_feats = _date_features(df)
    artist_feats = _artist_features(df)
    dim_feats = _dimension_features(df)
    med_feats, vec = _medium_features(df)

    feature_df = pd.concat([date_feats, artist_feats, dim_feats, med_feats], axis=1)

    # ── Encode target ────────────────────────────────────────────────────
    le = LabelEncoder()
    target_encoded = le.fit_transform(df[TARGET_COL].astype(str))
    target_series = pd.Series(target_encoded, name="target", index=df.index)

    clf_df = pd.concat([feature_df, target_series], axis=1)
    clust_df = feature_df.copy()

    print(
        f"  feature matrix : {feature_df.shape[0]:,} samples × {feature_df.shape[1]} features"
    )
    print(f"  target classes : {le.classes_.tolist()}")

    meta = {
        "label_encoder": le,
        "tfidf_vectorizer": vec,
        "feature_names": list(feature_df.columns),
        "class_names": list(le.classes_),
    }
    return clf_df, clust_df, meta


def scale_features(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """Fit a StandardScaler and return (X_scaled, scaler)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
