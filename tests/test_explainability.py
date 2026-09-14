"""
Unit tests for TreeSHAP explainability and patient risk attribution.
"""

import pytest
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from src.explainability import get_tree_explainer, compute_sample_shap_values, generate_local_explanation


@pytest.fixture
def fitted_tree_model():
    """Returns a small fitted XGBoost model for explainability testing."""
    np.random.seed(42)
    n = 60
    p = 4
    feature_names = ["BMI", "HighBP", "GenHlth", "Age"]
    X = pd.DataFrame(np.random.randn(n, p), columns=feature_names)
    y = np.random.choice([0, 1, 2], size=n)

    model = XGBClassifier(n_estimators=10, max_depth=3, objective="multi:softprob", num_class=3, random_state=42)
    model.fit(X, y)
    return model, X


def test_tree_explainer_and_attribution(fitted_tree_model):
    """Validates SHAP explainer initialization and patient explanation extraction."""
    model, X = fitted_tree_model
    explainer = get_tree_explainer(model)
    assert explainer is not None

    patient = X.iloc[0]
    explanation = generate_local_explanation(explainer, patient, target_class=2, top_n=3)

    assert "base_value" in explanation
    assert "top_contributions" in explanation
    assert "risk_factors" in explanation
    assert "protective_factors" in explanation
    assert len(explanation["top_contributions"]) <= 3
