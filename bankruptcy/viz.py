"""ROC curves and confusion matrices for report and README."""

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve

from bankruptcy.config import CLASSIFICATION_THRESHOLD, FIGURES_DIR


def plot_roc_comparison(y_test, y_proba_baseline_rf, y_proba_tuned_rf, y_proba_tuned_xgb):
    """Three-way ROC comparison on the 5th-year test set."""
    fpr_rf_base, tpr_rf_base, _ = roc_curve(y_test, y_proba_baseline_rf)
    roc_auc_rf_base = auc(fpr_rf_base, tpr_rf_base)

    fpr_rf_tuned, tpr_rf_tuned, _ = roc_curve(y_test, y_proba_tuned_rf)
    roc_auc_rf_tuned = auc(fpr_rf_tuned, tpr_rf_tuned)

    fpr_xgb_tuned, tpr_xgb_tuned, _ = roc_curve(y_test, y_proba_tuned_xgb)
    roc_auc_xgb_tuned = auc(fpr_xgb_tuned, tpr_xgb_tuned)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        fpr_rf_base,
        tpr_rf_base,
        label=f"Baseline RF (AUC = {roc_auc_rf_base:.3f})",
        linestyle="--",
        linewidth=1.5,
    )
    ax.plot(
        fpr_rf_tuned,
        tpr_rf_tuned,
        label=f"Tuned RF (AUC = {roc_auc_rf_tuned:.3f})",
        linewidth=2,
    )
    ax.plot(
        fpr_xgb_tuned,
        tpr_xgb_tuned,
        label=f"Tuned XGBoost (AUC = {roc_auc_xgb_tuned:.3f})",
        linewidth=2,
    )
    ax.plot([0, 1], [0, 1], "k:", label="Random chance")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        "ROC Curves: Baseline RF vs Tuned RF vs Tuned XGBoost (Test Set)", fontsize=14
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    return fig, {
        "baseline_rf": roc_auc_rf_base,
        "tuned_rf": roc_auc_rf_tuned,
        "tuned_xgb": roc_auc_xgb_tuned,
    }


def plot_confusion_matrix(y_test, y_proba, title):
    """Confusion matrix at the default classification threshold."""
    y_pred = (y_proba >= CLASSIFICATION_THRESHOLD).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-bankrupt (0)", "Bankrupt (1)"],
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"{title} (threshold={CLASSIFICATION_THRESHOLD})", fontsize=14)
    fig.tight_layout()
    return fig, cm


def save_figures(y_test, y_proba_baseline_rf, y_proba_tuned_rf, y_proba_tuned_xgb):
    """Write PNG figures for the report and README."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    roc_fig, _ = plot_roc_comparison(
        y_test, y_proba_baseline_rf, y_proba_tuned_rf, y_proba_tuned_xgb
    )
    roc_fig.savefig(
        FIGURES_DIR / "roc_curves_baseline_rf_tuned_rf_xgb.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(roc_fig)

    rf_cm_fig, _ = plot_confusion_matrix(y_test, y_proba_tuned_rf, "Tuned RF Confusion Matrix")
    rf_cm_fig.savefig(FIGURES_DIR / "confusion_matrix_rf.png", dpi=150, bbox_inches="tight")
    plt.close(rf_cm_fig)

    xgb_cm_fig, _ = plot_confusion_matrix(
        y_test, y_proba_tuned_xgb, "Tuned XGBoost Confusion Matrix"
    )
    xgb_cm_fig.savefig(
        FIGURES_DIR / "confusion_matrix_xgb.png", dpi=150, bbox_inches="tight"
    )
    plt.close(xgb_cm_fig)

    print(f"Figures saved to {FIGURES_DIR.resolve()}")
