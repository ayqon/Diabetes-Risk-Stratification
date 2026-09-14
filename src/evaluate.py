"""
Multi-class performance metrics and comparative evaluation utilities.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    precision_score, confusion_matrix
)


def evaluate_multiclass_performance(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes multi-class evaluation metrics with special focus on Macro-F1
    and per-class detection sensitivity across imbalanced classes.
    """
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "Model": name,
        "Accuracy": round(float(acc), 4),
        "Macro-F1": round(float(f1_macro), 4),
        "Weighted-F1": round(float(f1_weighted), 4),
        "No-DM Recall (0)": round(float(rec_per_class[0]), 4),
        "Pre-DM Recall (1)": round(float(rec_per_class[1]), 4) if len(rec_per_class) > 1 else 0.0,
        "DM Recall (2)": round(float(rec_per_class[2]), 4) if len(rec_per_class) > 2 else 0.0,
        "No-DM Precision (0)": round(float(prec_per_class[0]), 4),
        "Pre-DM Precision (1)": round(float(prec_per_class[1]), 4) if len(prec_per_class) > 1 else 0.0,
        "DM Precision (2)": round(float(prec_per_class[2]), 4) if len(prec_per_class) > 2 else 0.0,
        "Confusion_Matrix": cm.tolist(),
        "y_pred": y_pred,
        "y_proba": y_proba
    }


def build_comparison_table(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generates comparison summary table sorted by Macro-F1 score."""
    records = [{
        "Model": r["Model"],
        "Accuracy": r["Accuracy"],
        "Macro-F1": r["Macro-F1"],
        "Weighted-F1": r["Weighted-F1"],
        "No-DM Recall (0)": r["No-DM Recall (0)"],
        "Pre-DM Recall (1)": r["Pre-DM Recall (1)"],
        "DM Recall (2)": r["DM Recall (2)"]
    } for r in results_list]

    df = pd.DataFrame(records)
    return df.sort_values("Macro-F1", ascending=False).reset_index(drop=True)
