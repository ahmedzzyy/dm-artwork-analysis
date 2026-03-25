"""
predictor.py — inference engine for the Streamlit app.

Loads the trained RF model, scaler, and TF-IDF vectorizer from disk,
then transforms raw user input into the same feature vector the model
was trained on, and returns a prediction with per-class probabilities.

This is deliberately kept separate from the training code so the app
never needs to import sklearn training utilities or touch the CSVs.
"""

import numpy as np
import joblib

from src.config import MODELS_DIR


# ── Nationality list (top ones from the dataset) ───────────────────────────
# Used to populate the dropdown in the UI.
TOP_NATIONALITIES = [
    "American",
    "French",
    "German",
    "British",
    "Japanese",
    "Italian",
    "Swiss",
    "Spanish",
    "Austrian",
    "Dutch",
    "Russian",
    "Swedish",
    "Brazilian",
    "Canadian",
    "Belgian",
    "Danish",
    "Czech",
    "Hungarian",
    "Polish",
    "Argentine",
    "Mexican",
    "Australian",
    "Finnish",
    "Norwegian",
    "Chinese",
    "Unknown",
]

DEPARTMENTS = [
    "Architecture & Design",
    "Drawings & Prints",
    "Film",
    "Fluxus Collection",
    "Media and Performance",
    "Painting & Sculpture",
    "Photography",
]


# ── Model loader (cached — only loads once) ───────────────────────────────


class MoMAPredictor:
    """
    Wraps the trained model + all preprocessing artifacts needed
    to go from raw user input → prediction.

    Mirrors exactly what features.py builds during training:
      date features → artist features → dimension features → TF-IDF medium
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.meta = None  # contains label_encoder + tfidf_vectorizer
        self._loaded = False

    def load(self):
        """Load all artifacts from outputs/models/. Call once at app startup."""
        if self._loaded:
            return self

        # Find best classifier — filename pattern: best_classifier_<name>.pkl
        candidates = list(MODELS_DIR.glob("best_classifier_*.pkl"))
        if not candidates:
            raise FileNotFoundError(
                f"No trained model found in {MODELS_DIR}.\n"
                "Run  uv run python main.py  first to train and save the model."
            )
        self.model = joblib.load(candidates[0])

        scaler_path = MODELS_DIR / "feature_scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        self.scaler = joblib.load(scaler_path)

        meta_path = MODELS_DIR / "feature_meta.pkl"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Feature meta not found at {meta_path}.\n"
                "Run  uv run python main.py  first."
            )
        self.meta = joblib.load(meta_path)
        self._loaded = True
        return self

    # ── Feature construction ───────────────────────────────────────────────

    def _build_feature_vector(
        self,
        medium: str,
        nationality: str,
        creation_year: int,
        acquisition_year: int,
        is_female: bool,
        height_cm: float,
        width_cm: float,
    ) -> np.ndarray:
        """
        Reproduce the exact feature vector that features.py builds.
        Order must match: date → artist → dimension → TF-IDF medium.
        """
        vec = self.meta["tfidf_vectorizer"]
        le = self.meta["label_encoder"]

        # ── Date features ────────────────────────────────────────────────
        creation_decade = (creation_year // 10) * 10
        years_to_acquire = acquisition_year - creation_year

        # ── Artist features ──────────────────────────────────────────────
        nat_freq = self.meta.get("nationality_freq", {})
        nationality_enc = nat_freq.get(nationality, 0.0)
        career_len = 0.0  # unknown for a new prediction

        # ── Dimension features ───────────────────────────────────────────
        area_cm2 = height_cm * width_cm if (height_cm and width_cm) else 0.0
        height_cm = height_cm or 0.0
        width_cm = width_cm or 0.0
        area_cm2 = area_cm2 or 0.0

        # ── TF-IDF medium ────────────────────────────────────────────────
        tfidf_vec = vec.transform([medium.lower()]).toarray()[0]

        # ── Assemble in training order ───────────────────────────────────
        structured = np.array(
            [
                creation_year,
                creation_decade,
                acquisition_year,
                years_to_acquire,
                float(is_female),
                career_len,
                nationality_enc,
                height_cm,
                width_cm,
                area_cm2,
            ],
            dtype=float,
        )

        feature_vector = np.concatenate([structured, tfidf_vec])
        return feature_vector

    # ── Public predict API ─────────────────────────────────────────────────

    def predict(
        self,
        medium: str,
        nationality: str,
        creation_year: int,
        acquisition_year: int,
        is_female: bool = False,
        height_cm: float = 0.0,
        width_cm: float = 0.0,
    ) -> dict:
        """
        Returns a dict:
          predicted_class   : str   department name
          confidence        : float  probability of top class
          probabilities     : dict   {department: probability}
          top_features      : list   [(feature_name, importance), ...]
        """
        fv = self._build_feature_vector(
            medium,
            nationality,
            creation_year,
            acquisition_year,
            is_female,
            height_cm,
            width_cm,
        )
        fv_scaled = self.scaler.transform(fv.reshape(1, -1))

        proba = self.model.predict_proba(fv_scaled)[0]
        class_idx = int(np.argmax(proba))
        le = self.meta["label_encoder"]
        classes = le.classes_

        prob_dict = {cls: float(p) for cls, p in zip(classes, proba)}
        predicted = classes[class_idx]

        # Feature importances for this specific prediction
        # (global RF importances — we weight by the input feature values)
        feature_names = self.meta["feature_names"]
        importances = self.model.feature_importances_
        fv_abs = np.abs(fv_scaled[0])
        weighted = importances * fv_abs
        top_idx = np.argsort(weighted)[::-1][:10]
        top_features = [(feature_names[i], float(weighted[i])) for i in top_idx]

        return {
            "predicted_class": predicted,
            "confidence": float(proba[class_idx]),
            "probabilities": prob_dict,
            "top_features": top_features,
        }


# ── Singleton — one instance shared across Streamlit reruns ───────────────

_predictor_instance = None


def get_predictor() -> MoMAPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = MoMAPredictor().load()
    return _predictor_instance
