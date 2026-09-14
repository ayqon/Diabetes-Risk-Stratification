"""
Explainability and clinical decision support utilities using SHAP (Shapley Additive Explanations).
Provides global and patient-level attribution for multi-class tree models.
"""

from typing import List, Optional, Dict, Any, Union
import numpy as np
import pandas as pd
import shap


def get_tree_explainer(model: Any) -> shap.TreeExplainer:
    """Initializes TreeExplainer for tree-based ensemble models (XGBoost, Random Forest)."""
    return shap.TreeExplainer(model)


def compute_sample_shap_values(
    explainer: shap.TreeExplainer,
    X_sample: pd.DataFrame
) -> Union[np.ndarray, List[np.ndarray]]:
    """Computes SHAP values on a provided DataFrame sample."""
    return explainer.shap_values(X_sample)


def generate_local_explanation(
    explainer: shap.TreeExplainer,
    patient_series: pd.Series,
    target_class: int = 2,
    top_n: int = 8
) -> Dict[str, Any]:
    """
    Generates structured local attribution explanation for an individual patient.

    Args:
        explainer: Fitted SHAP TreeExplainer.
        patient_series: Feature values for a single patient.
        target_class: Target class index to explain (0: No Diabetes, 1: Prediabetes, 2: Diabetes).
        top_n: Number of top contributing features to return.

    Returns:
        Dict[str, Any]: Top positive (risk increasing) and negative (risk decreasing) feature impacts.
    """
    sample_df = pd.DataFrame([patient_series])
    shap_vals = explainer.shap_values(sample_df)

    if isinstance(shap_vals, list):
        # List of arrays per class: [class_0, class_1, class_2]
        cls_idx = min(target_class, len(shap_vals) - 1)
        shap_array = shap_vals[cls_idx][0]
    elif hasattr(shap_vals, 'ndim') and shap_vals.ndim == 3:
        # Array of shape (n_samples, n_features, n_classes)
        cls_idx = min(target_class, shap_vals.shape[2] - 1)
        shap_array = shap_vals[0, :, cls_idx]
    elif hasattr(shap_vals, 'ndim') and shap_vals.ndim == 2:
        shap_array = shap_vals[0]
    else:
        shap_array = np.array(shap_vals).flatten()

    feature_impacts = pd.Series(shap_array, index=patient_series.index)
    sorted_impacts = feature_impacts.sort_values(key=abs, ascending=False)

    top_features = sorted_impacts.head(top_n).to_dict()

    expected_val = explainer.expected_value
    if isinstance(expected_val, (list, np.ndarray)):
        cls_idx = min(target_class, len(expected_val) - 1)
        base_val = float(expected_val[cls_idx])
    else:
        base_val = float(expected_val)

    return {
        'target_class': target_class,
        'base_value': round(base_val, 4),
        'top_contributions': {k: round(float(v), 4) for k, v in top_features.items()},
        'risk_factors': {k: round(float(v), 4) for k, v in top_features.items() if v > 0},
        'protective_factors': {k: round(float(v), 4) for k, v in top_features.items() if v < 0}
    }
