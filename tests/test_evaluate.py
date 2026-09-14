"""
Unit tests for multi-class performance evaluation and comparison table building.
"""

import pytest
import numpy as np
import pandas as pd
from src.evaluate import evaluate_multiclass_performance, build_comparison_table


def test_evaluate_multiclass_performance():
    """Validates multi-class precision, recall, and Macro-F1 metric calculation."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 1, 0, 2, 2, 2, 1])

    res = evaluate_multiclass_performance("TestModel", y_true, y_pred)

    assert res["Model"] == "TestModel"
    assert 0.0 <= res["Accuracy"] <= 1.0
    assert 0.0 <= res["Macro-F1"] <= 1.0
    assert 0.0 <= res["Weighted-F1"] <= 1.0
    assert 0.0 <= res["No-DM Recall (0)"] <= 1.0
    assert 0.0 <= res["Pre-DM Recall (1)"] <= 1.0
    assert 0.0 <= res["DM Recall (2)"] <= 1.0
    assert isinstance(res["Confusion_Matrix"], list)
    assert len(res["Confusion_Matrix"]) == 3


def test_build_comparison_table():
    """Validates multi-model summary table generation and sorting by Macro-F1."""
    m1 = {
        "Model": "Model_A", "Accuracy": 0.80, "Macro-F1": 0.45, "Weighted-F1": 0.78,
        "No-DM Recall (0)": 0.90, "Pre-DM Recall (1)": 0.05, "DM Recall (2)": 0.40
    }
    m2 = {
        "Model": "Model_B", "Accuracy": 0.84, "Macro-F1": 0.52, "Weighted-F1": 0.82,
        "No-DM Recall (0)": 0.92, "Pre-DM Recall (1)": 0.12, "DM Recall (2)": 0.55
    }

    table = build_comparison_table([m1, m2])
    assert len(table) == 2
    assert table.iloc[0]["Model"] == "Model_B"  # Higher Macro-F1 first
