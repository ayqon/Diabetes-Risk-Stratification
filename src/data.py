"""
Data loading, validation, preprocessing, and stratified partitioning for CDC Diabetes dataset.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_diabetes_data(file_path: str = "dataset/diabetes_health_indicators.csv") -> pd.DataFrame:
    """Loads and validates the CDC Diabetes Health Indicators dataset."""
    df = pd.read_csv(file_path)
    if "Diabetes_012" not in df.columns:
        raise ValueError("Dataset must contain target column 'Diabetes_012'.")
    return df


def split_and_scale_data(
    df: pd.DataFrame,
    target_col: str = "Diabetes_012",
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Splits CDC dataset into stratified train and test partitions (80/20).
    StandardScaler is fitted strictly on the training partition.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    feature_names = X.columns.tolist()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=feature_names)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=feature_names)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        scaler
    )


def get_feature_metadata() -> Dict[str, str]:
    """Returns human-readable clinical descriptions for CDC BRFSS attributes."""
    return {
        "HighBP": "High Blood Pressure history (0=No, 1=Yes)",
        "HighChol": "High Cholesterol diagnosed (0=No, 1=Yes)",
        "CholCheck": "Cholesterol check in past 5 years (0=No, 1=Yes)",
        "BMI": "Body Mass Index (kg/m^2)",
        "Smoker": "Smoked at least 100 cigarettes in lifetime (0=No, 1=Yes)",
        "Stroke": "History of stroke (0=No, 1=Yes)",
        "HeartDiseaseorAttack": "Coronary heart disease or myocardial infarction (0=No, 1=Yes)",
        "PhysActivity": "Physical activity in past 30 days (0=No, 1=Yes)",
        "Fruits": "Consume fruit 1+ times per day (0=No, 1=Yes)",
        "Veggies": "Consume vegetables 1+ times per day (0=No, 1=Yes)",
        "HvyAlcoholConsump": "Heavy alcohol consumption (0=No, 1=Yes)",
        "AnyHealthcare": "Has any health insurance/coverage (0=No, 1=Yes)",
        "NoDocbcCost": "Could not see doctor due to cost in past 12mo (0=No, 1=Yes)",
        "GenHlth": "Self-rated general health (1=Excellent to 5=Poor)",
        "MentHlth": "Days of poor mental health in past 30 days (0-30)",
        "PhysHlth": "Days of physical illness/injury in past 30 days (0-30)",
        "DiffWalk": "Serious difficulty walking or climbing stairs (0=No, 1=Yes)",
        "Sex": "Biological sex (0=Female, 1=Male)",
        "Age": "Age tier category (1=18-24 to 13=80+)",
        "Education": "Highest education level completed (1-6 scale)",
        "Income": "Household income tier category (1-8 scale)"
    }
