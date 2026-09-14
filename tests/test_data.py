"""
Unit tests for data loading, metadata, and stratified train/test partitioning.
"""

import os
import pytest
import pandas as pd
import numpy as np
from src.data import load_diabetes_data, split_and_scale_data, get_feature_metadata


@pytest.fixture
def mock_diabetes_df():
    """Generates synthetic dataframe matching CDC BRFSS structure for fast testing."""
    np.random.seed(42)
    n = 200
    features = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
        "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
    ]
    data = {f: np.random.choice([0, 1], size=n) for f in features}
    data["BMI"] = np.random.uniform(18, 45, size=n)
    data["GenHlth"] = np.random.choice([1, 2, 3, 4, 5], size=n)
    data["Age"] = np.random.choice(range(1, 14), size=n)
    data["Diabetes_012"] = np.random.choice([0, 1, 2], size=n, p=[0.80, 0.05, 0.15])
    return pd.DataFrame(data)


def test_get_feature_metadata():
    """Validates that metadata dict contains all 21 CDC features."""
    meta = get_feature_metadata()
    assert isinstance(meta, dict)
    assert len(meta) == 21
    assert "HighBP" in meta
    assert "BMI" in meta
    assert "GenHlth" in meta


def test_split_and_scale_data(mock_diabetes_df):
    """Validates split proportions, scaling properties, and target preservation."""
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(
        mock_diabetes_df,
        target_col="Diabetes_012",
        test_size=0.20,
        random_state=42
    )

    assert len(X_train) == 160
    assert len(X_test) == 40
    assert len(y_train) == 160
    assert len(y_test) == 40
    assert X_train.shape[1] == 21
    assert X_test.shape[1] == 21

    # Scaled training data should have approximately mean 0
    assert np.allclose(X_train.mean().values, 0, atol=1e-1)


def test_load_diabetes_data_invalid():
    """Validates error raising when target column is absent."""
    df_invalid = pd.DataFrame({"FeatureA": [1, 2], "FeatureB": [3, 4]})
    temp_path = "tests/temp_test.csv"
    df_invalid.to_csv(temp_path, index=False)

    try:
        with pytest.raises(ValueError, match="Diabetes_012"):
            load_diabetes_data(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
