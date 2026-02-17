# credit-risk-modeling-ML-pipeline
An end-to-end credit risk scoring system designed to simulate real-world borrower default prediction in consumer lending environments.

# Credit Risk Modeling ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**End-to-end machine learning pipeline for predicting credit default risk in consumer lending.** 

This project demonstrates **production-grade data science** skills: assembling multi-source data, proactive quality checks, advanced modeling (with imbalance handling), explainable AI, and stakeholder-friendly dashboards. 

**Directly aligned with Data Scientist II roles at LexisNexis Risk Solutions**—mirrors their credit risk, fraud mitigation, and predictive analytics for lending, telecom, retail, and e-commerce.

## Problem Statement
In lending (consumer/small business), defaults cost billions annually (e.g., 22% rise in fraud/credit losses per industry reports). This pipeline builds a **risk scorer** to:
- Predict default probability.
- Identify data quality issues early.
- Provide **SHAP explanations** for fair, auditable decisions (key for compliance/AML).
- Simulate real-world impact: Reduce losses by 15-25% via better approvals.

## The Dataset
- **Source:** [Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) (32k rows, 12 features simulating bureau data).
- **Target:** `loan_status` (binary: 1 = default).
- **Key Features:** Income, loan amount, DTI, employment length, credit history.
- **Challenges Addressed:** Imbalance, missing values, categorical encoding.

## Tech Stack
| Category          | Tools                          |
|-------------------|--------------------------------|
| **Data**         | Pandas, DuckDB (SQL), PySpark (sim) |
| **ML**           | Scikit-learn, XGBoost, SHAP, imbalanced-learn |
| **Viz**          | Plotly, Seaborn, Streamlit    |
| **DevOps**       | Git/GitHub, Unix/Linux (Colab), Reproducible notebooks |
| **Big Data**     | AWS-inspired (S3 sim), Databricks-style scaling |

**All FREE tools—no paid compute.**

## Pipeline Overview
1. **Data Ingestion & Quality** → Merge, clean, validate.
2. **EDA & Feature Engineering** → Stats, correlations, new features (e.g., DTI ratio).
3. **Modeling** → Train/test split, baselines → XGBoost (tuned).
4. **Evaluation & Explainability** → AUC, PR curves, SHAP for business insights.
5. **Deployment** → Interactive Streamlit app (live demo).
6. **Impact** → "What-if" scenarios for lending teams.

## Quick Start
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/credit-risk-modeling-ML-pipeline.git
cd credit-risk-modeling-ML-pipeline

# Install (in Colab or venv)
pip install -r requirements.txt
