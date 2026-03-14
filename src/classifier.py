"""
classifier.py — supervised learning on artwork metadata.

Three models: Logistic Regression (baseline), Random Forest, SVM.
Evaluation: accuracy, classification report, confusion matrix.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.config import (
    TARGET_COL,
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    SVM_C,
    SVM_KERNEL,
    LR_MAX_ITER,
    PLOT_STYLE,
    FIGSIZE_SQ,
    FIGSIZE_WIDE,
)
from src.utils import save_fig, save_model, timer, print_section

plt.style.use(PLOT_STYLE)


# ── Data prep ──────────────────────────────────────────────────────────────


def prepare_data(clf_df: pd.DataFrame) -> tuple:
    """
    Split feature matrix into train/test sets and scale.

    Returns X_train, X_test, y_train, y_test, scaler, feature_names
    """
    feature_cols = [c for c in clf_df.columns if c != "target"]
    X = clf_df[feature_cols].values
    y = clf_df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(
        f"  train: {X_train.shape[0]:,}  test: {X_test.shape[0]:,}  features: {X_train.shape[1]}"
    )
    return X_train, X_test, y_train, y_test, scaler, feature_cols


# ── Model trainers ─────────────────────────────────────────────────────────


@timer
def train_logistic(X_train, y_train) -> LogisticRegression:
    """Logistic Regression — fast baseline."""
    model = LogisticRegression(
        max_iter=LR_MAX_ITER,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        C=1.0,
    )
    model.fit(X_train, y_train)
    return model


@timer
def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Random Forest — our main model. Also gives feature importances."""
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


@timer
def train_svm(X_train, y_train) -> SVC:
    """SVM — good on high-dimensional TF-IDF space. Slower on large datasets."""
    # For large datasets, subsample to keep training feasible
    if len(X_train) > 20_000:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_train), 20_000, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
        print("    (SVM: subsampled to 20k for speed)")

    model = SVC(
        C=SVM_C,
        kernel=SVM_KERNEL,
        probability=True,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


# ── Evaluation ─────────────────────────────────────────────────────────────


def evaluate(model, X_test, y_test, class_names: list, label: str) -> dict:
    """
    Predict, compute metrics, plot confusion matrix.
    Returns a metrics dict.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  [{label}]  accuracy = {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(f"Confusion Matrix — {label}", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, f"confusion_{label.lower().replace(' ', '_')}", subdir="figures")
    plt.close(fig)

    return {
        "label": label,
        "model": model,
        "accuracy": acc,
        "report": report,
        "cm": cm,
    }


def cross_validate_model(model, X_train, y_train, label: str) -> np.ndarray:
    """Quick CV on the training set to check for over/underfitting."""
    scores = cross_val_score(
        model, X_train, y_train, cv=CV_FOLDS, scoring="accuracy", n_jobs=-1
    )
    print(f"  [{label}] CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


# ── Visualisations ─────────────────────────────────────────────────────────


def plot_model_comparison(results: list[dict]) -> plt.Figure:
    """Bar chart comparing test accuracies of all models."""
    labels = [r["label"] for r in results]
    accs = [r["accuracy"] for r in results]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        labels, accs, color=colors[: len(labels)], width=0.5, edgecolor="white"
    )
    ax.set_ylim(0, min(1.0, max(accs) + 0.15))
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Model Comparison — Test Accuracy", fontweight="bold")

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    fig.tight_layout()
    save_fig(fig, "model_comparison", subdir="figures")
    return fig


def plot_feature_importances(
    rf_model: RandomForestClassifier, feature_names: list, top_n: int = 25
) -> plt.Figure:
    """Horizontal bar chart of top-N feature importances from the RF."""
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.3)))
    colors = ["#C44E52" if n.startswith("med_") else "#4C72B0" for n in top.index]
    ax.barh(top.index, top.values, color=colors)
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title(f"Top {top_n} Feature Importances — Random Forest", fontweight="bold")

    # Simple legend
    from matplotlib.patches import Patch

    legend = [
        Patch(color="#C44E52", label="TF-IDF (medium)"),
        Patch(color="#4C72B0", label="Structured feature"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8)

    fig.tight_layout()
    save_fig(fig, "feature_importances", subdir="figures")
    return fig


# ── Orchestrator ───────────────────────────────────────────────────────────


@timer
def run_classification(clf_df: pd.DataFrame, meta: dict) -> tuple[dict, list]:
    """
    Full classification pipeline.

    Returns
    -------
    best_result : dict with best model + metrics
    all_results : list of result dicts for all models
    """
    print_section("Classification")

    X_train, X_test, y_train, y_test, scaler, feat_cols = prepare_data(clf_df)
    class_names = meta["class_names"]

    # ── Train ────────────────────────────────────────────────────────────
    print("\nTraining Logistic Regression...")
    lr = train_logistic(X_train, y_train)

    print("\nTraining Random Forest...")
    rf = train_random_forest(X_train, y_train)

    print("\nTraining SVM...")
    svm = train_svm(X_train, y_train)

    # ── Evaluate ─────────────────────────────────────────────────────────
    print("\nEvaluating...")
    all_results = [
        evaluate(lr, X_test, y_test, class_names, "Logistic Regression"),
        evaluate(rf, X_test, y_test, class_names, "Random Forest"),
        evaluate(svm, X_test, y_test, class_names, "SVM"),
    ]

    # ── Plots ────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    plt.close(plot_model_comparison(all_results))
    plt.close(plot_feature_importances(rf, feat_cols))

    # ── Save best model ──────────────────────────────────────────────────
    best = max(all_results, key=lambda r: r["accuracy"])
    print(f"\n✓ Best model: {best['label']}  (accuracy={best['accuracy']:.4f})")
    save_model(
        best["model"], f"best_classifier_{best['label'].lower().replace(' ', '_')}"
    )
    save_model(scaler, "feature_scaler")

    return best, all_results
