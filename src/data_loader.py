"""
data_loader.py — load the MoMA Artworks + Artists CSVs, merge them,
and return a clean DataFrame ready for feature engineering.
"""

import re
import warnings
import numpy as np
import pandas as pd

from src.config import (
    ARTWORKS_CSV,
    ARTISTS_CSV,
    TARGET_COL,
    MIN_CLASS_SAMPLES,
)
from src.utils import timer

warnings.filterwarnings("ignore")


# ── Raw loading ────────────────────────────────────────────────────────────


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read both CSVs with low_memory=False (mixed dtypes are common here).
    Returns (artworks_df, artists_df).
    """
    print("Loading CSVs...")
    artworks = pd.read_csv(ARTWORKS_CSV, low_memory=False)
    artists = pd.read_csv(ARTISTS_CSV, low_memory=False)
    print(f"  artworks : {artworks.shape[0]:,} rows × {artworks.shape[1]} cols")
    print(f"  artists  : {artists.shape[0]:,} rows × {artists.shape[1]} cols")
    return artworks, artists


def merge_datasets(artworks: pd.DataFrame, artists: pd.DataFrame) -> pd.DataFrame:
    """
    MoMA artworks have a ConstituentID column that links to artists.
    Some artworks have multiple artists (comma-separated IDs) — we take
    the *first* listed artist only to keep the join 1-to-1.
    """

    def extract_first_id(val):
        if pd.isna(val):
            return np.nan
        nums = re.findall(r"\d+", str(val))
        return int(nums[0]) if nums else np.nan

    artworks = artworks.copy()
    artworks["ConstituentID_first"] = artworks["ConstituentID"].apply(extract_first_id)

    artists_renamed = artists.rename(
        columns={
            "ConstituentID": "ConstituentID_first",
            "DisplayName": "ArtistName",
            "Nationality": "ArtistNationality",
            "Gender": "ArtistGender",
            "BeginDate": "ArtistBirthYear",
            "EndDate": "ArtistDeathYear",
        }
    )

    merged = artworks.merge(
        artists_renamed[
            [
                "ConstituentID_first",
                "ArtistName",
                "ArtistNationality",
                "ArtistGender",
                "ArtistBirthYear",
                "ArtistDeathYear",
            ]
        ],
        on="ConstituentID_first",
        how="left",
    )
    print(f"  merged   : {merged.shape[0]:,} rows × {merged.shape[1]} cols")
    return merged


# ── Cleaning helpers ───────────────────────────────────────────────────────


def _parse_year(series: pd.Series) -> pd.Series:
    """
    Extract a 4-digit year from messy strings like '1984', 'c. 1920',
    '1965-70', '(1888)'. Returns float (NaN where unparseable).
    """

    def _get_year(val):
        try:
            if pd.isna(val):
                return np.nan
            m = re.search(r"\b(1[5-9]\d{2}|20[0-2]\d)\b", str(val))
            return int(m.group()) if m else np.nan
        except Exception:
            return np.nan

    return series.apply(_get_year)


def _parse_dimensions(series: pd.Series) -> pd.DataFrame:
    """
    MoMA dimensions strings look like:
      '23 3/4 × 31 7/8" (60.3 × 81 cm)'
    Extract height_cm and width_cm from the metric part.
    Returns a 2-col DataFrame (same index as input).
    """

    def _extract(val):
        try:
            if pd.isna(val):
                return np.nan, np.nan
            cm_part = re.search(r"\(([^)]+cm[^)]*)\)", str(val))
            if not cm_part:
                return np.nan, np.nan
            # \d+\.?\d* requires at least one digit — never matches a bare "."
            nums = re.findall(r"\d+\.?\d*", cm_part.group())
            nums = [float(n) for n in nums if 0 < float(n) < 10_000]
            if len(nums) >= 2:
                return nums[0], nums[1]  # MoMA lists h × w
            return np.nan, np.nan
        except Exception:
            return np.nan, np.nan

    results = series.apply(_extract)
    dim_df = pd.DataFrame(
        results.tolist(), columns=["height_cm", "width_cm"], index=series.index
    )
    return dim_df


# ── Main cleaner ───────────────────────────────────────────────────────────


@timer
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline. Returns a tidy DataFrame."""
    df = df.copy()

    # ── Standardise all string columns ──────────────────────────────────
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "": np.nan, "Unknown": np.nan})

    # Helper: safely fetch a column or an all-NaN series of the same length
    def _col(name):
        return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

    # ── Parse dates ─────────────────────────────────────────────────────
    df["creation_year"] = _parse_year(_col("Date"))
    df["acquisition_year"] = _parse_year(_col("DateAcquired"))
    df["artist_birth_year"] = _parse_year(_col("ArtistBirthYear"))
    df["artist_death_year"] = _parse_year(_col("ArtistDeathYear"))

    df["creation_decade"] = (df["creation_year"] // 10 * 10).astype("Int64")
    df["years_to_acquire"] = df["acquisition_year"] - df["creation_year"]
    df["artist_career_len"] = df["artist_death_year"] - df["artist_birth_year"]

    # ── Parse dimensions ────────────────────────────────────────────────
    dim_df = _parse_dimensions(_col("Dimensions"))
    df = pd.concat([df, dim_df], axis=1)
    df["area_cm2"] = df["height_cm"] * df["width_cm"]

    # ── Gender → binary ─────────────────────────────────────────────────
    df["is_female"] = (
        _col("ArtistGender")
        .astype(str)
        .str.lower()
        .str.contains("female")
        .astype("Int8")
    )

    # ── Drop columns with >60% missing ──────────────────────────────────
    thresh = int(0.40 * len(df))
    df = df.dropna(axis=1, thresh=thresh)

    # ── Drop rows where target is missing ───────────────────────────────
    df = df.dropna(subset=[TARGET_COL])

    # ── Drop tiny classes ───────────────────────────────────────────────
    class_counts = df[TARGET_COL].value_counts()
    valid_classes = class_counts[class_counts >= MIN_CLASS_SAMPLES].index
    df = df[df[TARGET_COL].isin(valid_classes)].copy()

    # ── Reset index ─────────────────────────────────────────────────────
    df = df.reset_index(drop=True)

    print(f"  clean df : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  classes  : {df[TARGET_COL].nunique()} departments")
    return df


@timer
def load_and_clean() -> pd.DataFrame:
    """
    End-to-end convenience: load CSVs → merge → clean → return.
    This is the only function notebooks / main.py need to call.
    """
    artworks, artists = load_raw()
    merged = merge_datasets(artworks, artists)
    df = clean(merged)

    print("\nClass distribution:")
    print(df[TARGET_COL].value_counts().to_string())
    return df
