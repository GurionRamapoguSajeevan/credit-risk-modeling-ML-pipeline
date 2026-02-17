# credit-risk-modeling-ML-pipeline
An end-to-end **credit risk scoring prediction system** designed to simulate real-world **borrower default prediction in consumer lending environments**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://grs-credit-risk-modeling-ml-pipeline-cns3z2uh5bpbijwpcwespo.streamlit.app/) <!-- Replace with your actual deployed URL if available -->

# Credit Risk Modeling ML Pipeline
An end-to-end **credit risk scoring prediction system** designed to simulate real-world **borrower default prediction in consumer lending environments**.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project demonstrates **production-grade data science** skills: assembling multi-source data, proactive quality checks, advanced modeling (with imbalance handling), explainable AI, and stakeholder-friendly dashboards. 

**Directly aligned with Data Scientist work at LexisNexis Risk Solutions**—mirrors their credit risk, fraud mitigation, and predictive analytics for lending, telecom, retail, and e-commerce.

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
| **Big Data**     | Streamlit (public hosting) |

**All FREE tools—no paid compute.**

## Pipeline Overview
1. **Data Ingestion & Quality** → Load CSV, check missings/duplicates/outliers, impute (median), cap extremes.
2. **EDA & Feature Engineering** → Distributions/correlations, new features (DTI ratio, loan-to-income, emp-to-age stability).
3. **Preprocessing** → Train/test split (stratified), scaling (StandardScaler), imbalance handling (SMOTE).
4. **Modeling** → 
   - Logistic Regression (interpretable baseline).
   - Random Forest (robust ensemble).
   - XGBoost (gradient boosting — production-grade for risk data).
5. **Evaluation** → Risk-focused metrics (AUC-ROC, PR-AUC, confusion matrix), curves (ROC/PR).
6. **Explainability** → SHAP (global summary + local waterfalls).
7. **Deployment** → Interactive Streamlit app (inputs → prediction + SHAP "why").
6. **Impact** → "What-if" scenarios for lending teams.

## Quick Start
```bash
# Clone the repo
git clone https://github.com/GurionRamapoguSajeevan/credit-risk-modeling-ML-pipeline.git
cd credit-risk-modeling-ML-pipeline

# Install dependencies
pip install -r requirements.txt

# Run notebook locally (or open in Colab)
jupyter notebook Credit_risk_main_code.ipynb

# For Streamlit app (local)
streamlit run app.py
```

## 📈 Results

- **Best Model**: XGBoost → 0.9343 AUC-ROC | 0.8811 PR-AUC | 92% Accuracy.
- **Key Insight**: High DTI + renter status + D grade → 3x default risk.
- **Business Value**: Like LexisNexis RiskView™ — actionable for cross-functional teams (sales, compliance, ops). Reduces false positives by ~25% vs. baseline.

**Top Features (SHAP Impact):**
* person_home_ownership_RENT
* loan_int_rate
* person_income
* dti_ratio
* loan_grade_D

**SHAP Summary Plot**:
  <img width="789" height="940" alt="shap_dot(summary)_plot" src="https://github.com/user-attachments/assets/201a94f3-51ed-4db0-911e-d3056170e98a" />

**SHAP Bar Plot**:
  <img width="790" height="620" alt="shap_bar_plot" src="https://github.com/user-attachments/assets/fb6f7af7-15d1-4ab7-aa94-3d7d54fd7c38" />


**XGBoost ROC Curve**:
- (AUC 0.9343 — strong discrimination)

<img width="820" height="678" alt="image" src="https://github.com/user-attachments/assets/eb25f1f3-78d3-4bdb-91f6-ecc4f2b72fb0" />

## Project Structure: 
```
credit-risk-modeling-ML-pipeline/
├── Credit_risk_main_code.ipynb    # Full end-to-end notebook (EDA → models → SHAP)
├── app.py                         # Streamlit dashboard code
├── xgboost_credit_risk_model.pkl  # Saved XGBoost model
├── scaler.pkl                     # Saved scaler
├── shap_explainer.pkl             # Saved SHAP explainer
├── credit_risk_cleaned.csv        # Processed dataset (optional)
├── requirements.txt               # Dependencies
├── README.md
└── .gitignore
```

## Contributing to the project:
Contributions welcome! Fork the repo, create a branch, and submit a pull request. Focus on bug fixes, feature additions, or documentation improvements.

## License
**MIT** License. See [LICENSE](LICENSE) for details.

## Contact
- [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/GurionRamapoguSajeevan)
- [![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rs-gurion/)
- [![Email](https://img.shields.io/badge/email-%23D14836.svg?style=for-the-badge&logo=gmail&logoColor=white)](mailto:gurion7007@gmail.com)

**Built as a personal project** to brush up on and dive deep in Machine Learning for Credit Risk / fraud prediction; interview preparation purpose. Any feedback will be appreciated!
