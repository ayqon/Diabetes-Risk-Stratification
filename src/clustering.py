"""
Unsupervised patient phenotype discovery and clustering evaluation using K-Means.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def evaluate_kmeans_elbow_silhouette(
    X: pd.DataFrame,
    k_range: range = range(2, 7),
    sample_size: int = 25000,
    random_state: int = 42
) -> List[Dict[str, Any]]:
    """
    Evaluates inertia and silhouette scores across cluster counts K.
    Uses stratified/random subsampling for fast evaluation on large cohorts.
    """
    sample_len = min(sample_size, len(X))
    sample_idx = np.random.RandomState(random_state).choice(len(X), size=sample_len, replace=False)
    X_sample = X.iloc[sample_idx]

    metrics = []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=5, max_iter=200, random_state=random_state)
        labels = km.fit_predict(X_sample)
        sil = silhouette_score(X_sample, labels, metric="euclidean")

        metrics.append({
            "k": int(k),
            "inertia": float(km.inertia_),
            "silhouette": round(float(sil), 4)
        })

    return metrics


def fit_kmeans(
    X: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42
) -> Tuple[KMeans, np.ndarray]:
    """Fits final K-Means model on the feature matrix."""
    km = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=random_state
    )
    labels = km.fit_predict(X)
    return km, labels


def profile_patient_clusters(
    X_raw: pd.DataFrame,
    cluster_labels: np.ndarray,
    y_true: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Aggregates key clinical indicators per discovered patient cluster phenotype.
    """
    df_merged = X_raw.copy()
    df_merged["Cluster"] = cluster_labels
    if y_true is not None:
        df_merged["Diabetes_True"] = y_true.values

    agg_dict = {
        "BMI": "mean",
        "HighBP": lambda x: float((x == 1).mean() * 100),
        "HighChol": lambda x: float((x == 1).mean() * 100),
        "GenHlth": "mean",
        "Age": "mean",
        "PhysActivity": lambda x: float((x == 1).mean() * 100)
    }
    if y_true is not None:
        agg_dict["Diabetes_True"] = lambda x: float((x == 2).mean() * 100)

    profile = df_merged.groupby("Cluster").agg(agg_dict).round(2)
    col_names = ["Mean BMI", "High BP (%)", "High Chol (%)", "Gen Health (1-5)", "Mean Age Tier", "Phys Active (%)"]
    if y_true is not None:
        col_names.append("Diabetes Prevalence (%)")

    profile.columns = col_names
    return profile.reset_index()
