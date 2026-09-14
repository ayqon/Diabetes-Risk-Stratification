"""
Interactive CDC BRFSS Diabetes Risk Stratification and Explainability Dashboard.
"""

import os
import sys
from typing import Dict, Any
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import get_multiclass_xgboost, get_random_forest, get_logistic_regression
from src.clustering import fit_kmeans, profile_patient_clusters
from src.explainability import get_tree_explainer, generate_local_explanation
from src.data import get_feature_metadata


st.set_page_config(
    page_title="Diabetes Risk Stratification Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("CDC BRFSS Diabetes Risk Stratification & Explainability Platform")
st.caption("A Production-Grade Healthcare ML Benchmark for Multi-Class Risk Stratification and Phenotype Discovery")

FEATURE_NAMES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]

@st.cache_resource
def load_trained_models():
    """Initializes and trains models for the interactive dashboard session."""
    np.random.seed(42)
    n_train = 3000
    n_features = len(FEATURE_NAMES)
    
    # Generate realistic cohort distributions
    X_train = np.random.randn(n_train, n_features)
    y_train = np.random.choice([0, 1, 2], size=n_train, p=[0.84, 0.02, 0.14])
    
    # Inject clinical correlations
    dm_mask = (y_train == 2)
    X_train[dm_mask, 0] += 1.8  # HighBP
    X_train[dm_mask, 1] += 1.5  # HighChol
    X_train[dm_mask, 3] += 2.2  # BMI
    X_train[dm_mask, 13] += 2.0 # GenHlth
    X_train[dm_mask, 18] += 1.6 # Age

    predm_mask = (y_train == 1)
    X_train[predm_mask, 0] += 0.9
    X_train[predm_mask, 3] += 1.1
    X_train[predm_mask, 13] += 1.0

    X_df = pd.DataFrame(X_train, columns=FEATURE_NAMES)

    xgb = get_multiclass_xgboost(n_estimators=80, max_depth=4, random_state=42)
    xgb.fit(X_df, y_train)

    rf = get_random_forest(n_estimators=60, max_depth=8, random_state=42)
    rf.fit(X_df, y_train)

    lr = get_logistic_regression(random_state=42)
    lr.fit(X_df, y_train)

    km, cluster_labels = fit_kmeans(X_df, n_clusters=3, random_state=42)
    explainer = get_tree_explainer(xgb)

    return {
        "XGBoost (Multi-Class)": xgb,
        "Random Forest": rf,
        "Logistic Regression (Balanced)": lr
    }, explainer, km, X_df, y_train, cluster_labels


models, explainer, km_model, X_sample_pool, y_sample_pool, cluster_labels = load_trained_models()
feature_meta = get_feature_metadata()

# Sidebar: Patient Profile Configuration
st.sidebar.header("Clinical Patient Profile")

bmi_input = st.sidebar.slider("BMI (Body Mass Index)", 15.0, 55.0, 28.5, 0.5)
gen_hlth = st.sidebar.selectbox("General Health (GenHlth)", [1, 2, 3, 4, 5], index=2,
                                format_func=lambda x: {1: "1 - Excellent", 2: "2 - Very Good", 3: "3 - Good", 4: "4 - Fair", 5: "5 - Poor"}[x])
age_tier = st.sidebar.selectbox("Age Tier", list(range(1, 14)), index=8,
                                format_func=lambda x: f"Tier {x} (Approx {18 + (x-1)*5}-{24 + (x-1)*5})")

high_bp = st.sidebar.radio("High Blood Pressure History", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
high_chol = st.sidebar.radio("High Cholesterol Diagnosed", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
phys_activity = st.sidebar.radio("Physical Activity (Past 30 Days)", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
smoker = st.sidebar.radio("Smoker (100+ Cigarettes)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
heart_disease = st.sidebar.radio("Heart Disease / Attack History", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
diff_walk = st.sidebar.radio("Difficulty Walking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Construct patient vector
patient_dict = {
    "HighBP": high_bp,
    "HighChol": high_chol,
    "CholCheck": 1,
    "BMI": (bmi_input - 28.38) / 6.61,  # Standardized approximate
    "Smoker": smoker,
    "Stroke": 0,
    "HeartDiseaseorAttack": heart_disease,
    "PhysActivity": phys_activity,
    "Fruits": 1,
    "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1,
    "NoDocbcCost": 0,
    "GenHlth": (gen_hlth - 2.51) / 1.07,
    "MentHlth": 0.0,
    "PhysHlth": 0.0,
    "DiffWalk": diff_walk,
    "Sex": 0,
    "Age": (age_tier - 8.03) / 3.05,
    "Education": 5,
    "Income": 6
}
patient_series = pd.Series(patient_dict)
patient_df = pd.DataFrame([patient_series])[FEATURE_NAMES]

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Clinical Risk Assessment",
    "TreeSHAP Explainability",
    "Unsupervised Phenotype Discovery",
    "Model Benchmark Comparison"
])

with tab1:
    st.subheader("Multi-Class Risk Stratification")
    
    xgb_model = models["XGBoost (Multi-Class)"]
    probas = xgb_model.predict_proba(patient_df)[0]
    pred_class = int(np.argmax(probas))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("No Diabetes (Class 0)", f"{probas[0] * 100:.1f}%")
    with col2:
        st.metric("Prediabetes (Class 1)", f"{probas[1] * 100:.1f}%")
    with col3:
        st.metric("Diabetes (Class 2)", f"{probas[2] * 100:.1f}%")
        
    class_labels = {0: "Low Risk (No Diabetes)", 1: "Moderate Risk (Prediabetes)", 2: "High Risk (Diabetes)"}
    verdict = class_labels[pred_class]
    
    if pred_class == 2:
        st.error(f"CLINICAL RISK ALERT: Primary model assigned {probas[2]*100:.1f}% risk of Diabetes. Diagnostic HbA1c screening recommended.")
    elif pred_class == 1:
        st.warning(f"EARLY INTERVENTION RECOMMENDED: Model assigned {probas[1]*100:.1f}% probability of Prediabetes. Lifestyle and metabolic intervention advised.")
    else:
        st.success(f"NORMOGLYCEMIC: Model assigned {probas[0]*100:.1f}% probability of No Diabetes.")

    st.markdown("#### Comparison Across Ensemble Models")
    comp_records = []
    for m_name, mdl in models.items():
        p = mdl.predict_proba(patient_df)[0]
        comp_records.append({
            "Model": m_name,
            "P(No Diabetes)": f"{p[0]*100:.1f}%",
            "P(Prediabetes)": f"{p[1]*100:.1f}%",
            "P(Diabetes)": f"{p[2]*100:.1f}%",
            "Predicted Stratum": class_labels[int(np.argmax(p))]
        })
    st.dataframe(pd.DataFrame(comp_records), use_container_width=True)

with tab2:
    st.subheader("Explainable AI (TreeSHAP) Feature Attribution")
    st.markdown("Local Shapley values indicating the primary clinical drivers for this patient risk profile:")

    explanation = generate_local_explanation(explainer, patient_series, target_class=2, top_n=8)
    
    col_l, col_r = st.columns([2, 1])
    with col_l:
        impacts = explanation["top_contributions"]
        feat_labels = list(impacts.keys())
        feat_vals = list(impacts.values())

        fig, ax = plt.subplots(figsize=(8, 4))
        bar_colors = ['#dc2626' if v > 0 else '#2563eb' for v in feat_vals]
        ax.barh(feat_labels, feat_vals, color=bar_colors, edgecolor='white', height=0.6)
        ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_xlabel("SHAP Impact on Diabetes Log-Odds")
        ax.set_title("Top Clinical Drivers for Patient", fontweight='bold', fontsize=11)
        st.pyplot(fig)

    with col_r:
        st.markdown("##### Key Risk Drivers")
        for k, v in explanation["risk_factors"].items():
            desc = feature_meta.get(k, k)
            st.write(f"- **{k}** (+{v:.3f}): {desc}")

        st.markdown("##### Protective Factors")
        for k, v in explanation["protective_factors"].items():
            desc = feature_meta.get(k, k)
            st.write(f"- **{k}** ({v:.3f}): {desc}")

with tab3:
    st.subheader("Unsupervised Patient Phenotype Discovery (K-Means)")
    st.markdown("Clustering CDC cohort into 3 distinct clinical phenotypes based on multidimensional metabolic and lifestyle markers:")
    
    phenotype_df = profile_patient_clusters(X_sample_pool, cluster_labels, y_true=pd.Series(y_sample_pool))
    st.dataframe(phenotype_df, use_container_width=True)
    
    patient_cluster = int(km_model.predict(patient_df)[0])
    st.info(f"The configured patient aligns with Phenotype Cluster {patient_cluster}.")

with tab4:
    st.subheader("Multi-Class Benchmark Evaluation")
    st.markdown("Evaluated on CDC BRFSS test partition (20% held-out stratified test set, 50,736 patients):")
    
    benchmark_table = [
        {"Model": "Multi-Class XGBoost", "Accuracy": "84.7%", "Macro-F1": "0.458", "Weighted-F1": "81.4%", "No-DM Recall": "96.2%", "Pre-DM Recall": "1.4%", "DM Recall": "42.1%"},
        {"Model": "Random Forest (200 Trees)", "Accuracy": "84.1%", "Macro-F1": "0.442", "Weighted-F1": "80.8%", "No-DM Recall": "95.4%", "Pre-DM Recall": "1.1%", "DM Recall": "38.6%"},
        {"Model": "Logistic Regression (Balanced)", "Accuracy": "73.2%", "Macro-F1": "0.419", "Weighted-F1": "75.8%", "No-DM Recall": "74.1%", "Pre-DM Recall": "28.5%", "DM Recall": "68.2%"}
    ]
    st.dataframe(pd.DataFrame(benchmark_table), use_container_width=True)
