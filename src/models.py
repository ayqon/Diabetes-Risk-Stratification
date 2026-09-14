"""
Supervised multi-class classification model factories for diabetes risk prediction.
"""

from typing import Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_logistic_regression(
    class_weight: str = "balanced",
    max_iter: int = 1000,
    random_state: int = 42
) -> LogisticRegression:
    """Returns configured baseline Logistic Regression multi-class classifier."""
    return LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
        solver="lbfgs"
    )


def get_random_forest(
    n_estimators: int = 200,
    max_depth: int = 20,
    min_samples_split: int = 5,
    random_state: int = 42,
    n_jobs: int = -1
) -> RandomForestClassifier:
    """Returns configured Random Forest multi-class ensemble."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=n_jobs
    )


def get_multiclass_xgboost(
    n_estimators: int = 200,
    learning_rate: float = 0.08,
    max_depth: int = 6,
    random_state: int = 42,
    n_jobs: int = -1
) -> XGBClassifier:
    """Returns configured multi-class Gradient Boosted Decision Trees (XGBoost)."""
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=n_jobs
    )
