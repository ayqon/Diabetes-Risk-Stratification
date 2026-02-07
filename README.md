# Diabetes Prediction Analysis
FAIDM Individual Coursework - Student: U5750302

---

## Overview
This project was developed for the **Foundations of Artificial Intelligence and Data Mining (FAIDM)** module at the **University of Warwick**.

The objective of the assignment is to apply **clustering** and **classification** techniques to a real-world dataset and evaluate their performance.

The **CDC Diabetes Health Indicators dataset** is used to explore patterns in patient health data and predict diabetes status using machine learning models.

---

## Dataset
CDC Diabetes Health Indicators dataset (~253,000 records).

Target variable:

`Diabetes_012`

| Value | Class |
|------|------|
| 0 | No Diabetes |
| 1 | Prediabetes |
| 2 | Diabetes |

Main challenge: **class imbalance**, particularly for prediabetes cases.

---

## Workflow

| Step | Description |
|------|-------------|
| Data exploration | Dataset structure and class distribution |
| Preparation | Stratified train/test split and scaling |
| Clustering | K-Means with silhouette analysis |
| Classification | Logistic Regression and Random Forest |
| Evaluation | Accuracy, F1-score, confusion matrices |

---

## Models

| Model | Type | Purpose |
|------|------|---------|
| Logistic Regression | Linear classifier | Baseline model |
| Random Forest | Ensemble model | Advanced classifier |
| K-Means | Clustering | Pattern discovery |

---

## Evaluation Metrics

| Metric | Purpose |
|------|---------|
| Accuracy | Overall performance |
| Precision | Correct positive predictions |
| Recall | Detection ability |
| F1-score | Balance of precision and recall |
| Macro-F1 | Fair evaluation across classes |

---

## Results Summary
- K-Means clustering showed **weak but interpretable cluster separation**
- Logistic Regression provided a **baseline model**
- Random Forest achieved **improved predictive performance**
- Class imbalance affected **prediabetes detection**

---

## Repository Structure
```
Diabetes_Prediction_Analysis.ipynb
README.md
```

---

## Technologies
Python - Pandas - NumPy - Scikit-learn - Matplotlib - Seaborn

---

## Author
Ioannis Konstantinou  
U5750302
MSc Applied Artificial Intelligence  
University of Warwick
