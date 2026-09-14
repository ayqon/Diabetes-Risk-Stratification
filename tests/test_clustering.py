"""
Unit tests for K-Means unsupervised clustering and phenotype profiling.
"""

import pytest
import numpy as np
import pandas as pd
from src.clustering import evaluate_kmeans_elbow_silhouette, fit_kmeans, profile_patient_clusters


@pytest.fixture
def sample_feature_data():
    """Generates synthetic scaled clinical features for clustering."""
    np.random.seed(42)
    n = 150
    data = {
        "HighBP": np.random.choice([0, 1], size=n),
        "HighChol": np.random.choice([0, 1], size=n),
        "BMI": np.random.uniform(18, 40, size=n),
        "GenHlth": np.random.choice([1, 2, 3, 4, 5], size=n),
        "Age": np.random.choice(range(1, 14), size=n),
        "PhysActivity": np.random.choice([0, 1], size=n)
    }
    return pd.DataFrame(data)


def test_evaluate_kmeans_elbow_silhouette(sample_feature_data):
    """Validates computation of inertia and silhouette across k range."""
    results = evaluate_kmeans_elbow_silhouette(
        sample_feature_data,
        k_range=range(2, 5),
        sample_size=100,
        random_state=42
    )

    assert len(results) == 3
    for r in results:
        assert "k" in r
        assert "inertia" in r
        assert "silhouette" in r
        assert r["k"] in [2, 3, 4]
        assert r["inertia"] > 0
        assert -1.0 <= r["silhouette"] <= 1.0


def test_fit_kmeans(sample_feature_data):
    """Validates K-Means fitting and cluster label assignment."""
    km, labels = fit_kmeans(sample_feature_data, n_clusters=3, random_state=42)
    assert km.n_clusters == 3
    assert len(labels) == len(sample_feature_data)
    assert set(np.unique(labels)).issubset({0, 1, 2})


def test_profile_patient_clusters(sample_feature_data):
    """Validates cluster summary aggregation."""
    labels = np.random.choice([0, 1, 2], size=len(sample_feature_data))
    y_true = pd.Series(np.random.choice([0, 1, 2], size=len(sample_feature_data)))

    profile = profile_patient_clusters(sample_feature_data, labels, y_true=y_true)
    assert len(profile) == 3
    assert "Cluster" in profile.columns
    assert "Mean BMI" in profile.columns
    assert "High BP (%)" in profile.columns
    assert "Diabetes Prevalence (%)" in profile.columns
