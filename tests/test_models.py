"""
Unit tests for multi-class classification model factories and fitting.
"""

import pytest
import numpy as np
import pandas as pd
from src.models import get_logistic_regression, get_random_forest, get_multiclass_xgboost


@pytest.fixture
def dummy_train_data():
    """Generates small training dataset for testing model fit."""
    np.random.seed(42)
    n = 100
    p = 5
    X = pd.DataFrame(np.random.randn(n, p), columns=[f"feat_{i}" for i in range(p)])
    y = np.random.choice([0, 1, 2], size=n, p=[0.70, 0.10, 0.20])
    return X, y


def test_get_logistic_regression(dummy_train_data):
    """Validates baseline Logistic Regression instantiation and prediction."""
    X, y = dummy_train_data
    model = get_logistic_regression(random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    probas = model.predict_proba(X)

    assert len(preds) == len(y)
    assert probas.shape == (len(y), 3)


def test_get_random_forest(dummy_train_data):
    """Validates Random Forest multi-class ensemble."""
    X, y = dummy_train_data
    model = get_random_forest(n_estimators=10, max_depth=4, random_state=42, n_jobs=1)
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1, 2})


def test_get_multiclass_xgboost(dummy_train_data):
    """Validates Multi-Class XGBoost classifier."""
    X, y = dummy_train_data
    model = get_multiclass_xgboost(n_estimators=10, max_depth=3, random_state=42, n_jobs=1)
    model.fit(X, y)
    preds = model.predict(X)
    probas = model.predict_proba(X)

    assert len(preds) == len(y)
    assert probas.shape == (len(y), 3)
