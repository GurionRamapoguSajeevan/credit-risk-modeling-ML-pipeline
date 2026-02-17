import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ─── Page Config & Title ─────────────────────────────────────────────────
st.set_page_config(page_title="Credit Risk Scorer", layout="wide")
st.title("Credit Risk Scorer – XGBoost + SHAP Explainability")
st.info("""
**Quick facts about this tool**  
• Predicts the probability that a loan applicant might default  
• Trained on ~32,000 historical loan records  
• Uses XGBoost (strong predictive model) + SHAP (shows why the prediction was made)  
• **Not for real lending decisions** — educational / demonstration purpose only
""", icon="ℹ️")
st.markdown("""
Enter applicant information on the left.  
The model predicts default probability and shows **'WHY'** via SHAP (local explanation).  
Built to mirror risk assessment workflows (e.g., credit/fraud scoring).
""")

# ─── Load assets (cached) ────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model     = joblib.load('xgboost_credit_risk_model.pkl')
    scaler    = joblib.load('scaler.pkl')
    explainer = joblib.load('shap_explainer.pkl')
    return model, scaler, explainer

model, scaler, explainer = load_assets()

# ─── Sidebar Inputs ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Applicant / Loan Details")

    col1, col2 = st.columns(2)
    with col1:
        person_age = st.number_input("Age", 18, 90, value=30,
                                     help="Younger applicants sometimes carry slightly higher risk.")
        person_income = st.number_input(
            "Annual Income ($)", 5000, 500000, value=60000, step=1000,
            help="Higher income generally lowers risk (protective factor)."
        )
        person_emp_length = st.number_input("Employment Length (years)", 0.0, 40.0, value=5.0, step=0.5,
                                            help="Longer employment usually reduces risk.")
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)", 0, 50, value=8,
                                                     help="Longer history is generally protective.")

    with col2:
        loan_amnt = st.number_input("Loan Amount ($)", 1000, 50000, value=12000, step=500,
                                    help="Larger loans relative to income increase risk.")
        loan_int_rate = st.number_input(
            "Interest Rate (%)", 5.0, 30.0, value=12.0, step=0.1,
            help="Higher rates often indicate higher perceived risk by lenders."
        )
        loan_percent_income = st.slider("Loan as % of Income", 0.01, 1.0, value=0.20, step=0.01,
                                        help="Higher % → higher burden on income.")

    home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "OTHER"],
                                  help="Renters tend to have higher default rates than owners.")
    loan_intent = st.selectbox("Loan Intent", ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"],
                               help="Some purposes (e.g., venture, medical) can carry different risk profiles.")
    loan_grade = st.selectbox(
        "Loan Grade",
        ["A", "B", "C", "D", "E", "F", "G"],
        help="Grade D or worse is one of the strongest predictors of default."
    )
    default_on_file = st.checkbox("Has previous default on file?", value=False,
                                  help="Previous default significantly raises risk.")

    # Optional: show calculated DTI live in sidebar
    dti_ratio = loan_amnt / person_income if person_income > 0 else 0.0
    st.metric(
        "Calculated Debt-to-Income Ratio",
        f"{dti_ratio:.1%}",
        help="Ratio of loan amount to annual income — high values strongly increase risk."
    )

# ─── Derived features (must match training) ──────────────────────────────
dti_ratio      = loan_amnt / person_income if person_income > 0 else 0.0
loan_to_income = loan_amnt / person_income if person_income > 0 else 0.0
emp_to_age     = person_emp_length / person_age if person_age > 0 else 0.0

# ─── Create input row with ALL 25 columns ────────────────────────────────
input_dict = {
    'person_age':                   person_age,
    'person_income':                person_income,
    'person_emp_length':            person_emp_length,
    'loan_amnt':                    loan_amnt,
    'loan_int_rate':                loan_int_rate,
    'loan_percent_income':          loan_percent_income,
    'cb_person_cred_hist_length':   cb_person_cred_hist_length,
    'dti_ratio':                    dti_ratio,
    'loan_to_income':               loan_to_income,
    'emp_to_age':                   emp_to_age,

    # Home ownership dummies (MORTGAGE = all 0)
    'person_home_ownership_OTHER':  1 if home_ownership == "OTHER" else 0,
    'person_home_ownership_OWN':    1 if home_ownership == "OWN"   else 0,
    'person_home_ownership_RENT':   1 if home_ownership == "RENT"  else 0,

    # Loan intent dummies (DEBTCONSOLIDATION = all 0)
    'loan_intent_EDUCATION':        1 if loan_intent == "EDUCATION"        else 0,
    'loan_intent_HOMEIMPROVEMENT':  1 if loan_intent == "HOMEIMPROVEMENT"  else 0,
    'loan_intent_MEDICAL':          1 if loan_intent == "MEDICAL"          else 0,
    'loan_intent_PERSONAL':         1 if loan_intent == "PERSONAL"         else 0,
    'loan_intent_VENTURE':          1 if loan_intent == "VENTURE"          else 0,

    # Loan grade dummies (A = all 0)
    'loan_grade_B': 1 if loan_grade == "B" else 0,
    'loan_grade_C': 1 if loan_grade == "C" else 0,
    'loan_grade_D': 1 if loan_grade == "D" else 0,
    'loan_grade_E': 1 if loan_grade == "E" else 0,
    'loan_grade_F': 1 if loan_grade == "F" else 0,
    'loan_grade_G': 1 if loan_grade == "G" else 0,

    # Default flag
    'cb_person_default_on_file_Y': 1 if default_on_file else 0,
}

input_df = pd.DataFrame([input_dict])

# Reorder columns to exactly match training order
expected_columns = [
    'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
    'loan_percent_income', 'cb_person_cred_hist_length', 'dti_ratio', 'loan_to_income',
    'emp_to_age', 'person_home_ownership_OTHER', 'person_home_ownership_OWN',
    'person_home_ownership_RENT', 'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT',
    'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE',
    'loan_grade_B', 'loan_grade_C', 'loan_grade_D', 'loan_grade_E', 'loan_grade_F',
    'loan_grade_G', 'cb_person_default_on_file_Y'
]

input_df = input_df[expected_columns]

# Scale numeric columns (same as training)
numeric_cols = [
    'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
    'loan_percent_income', 'cb_person_cred_hist_length', 'dti_ratio', 'loan_to_income',
    'emp_to_age'
]
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# ─── Prediction & Explanation ────────────────────────────────────────────
if st.button("Calculate Risk Score", type="primary"):
    prob = model.predict_proba(input_df)[0][1]
    risk_band = "Low" if prob < 0.20 else "Medium" if prob < 0.50 else "High"

    left, right = st.columns([1, 2.5])

    with left:
        st.metric("Predicted Default Probability", f"{prob:.1%}", delta=None)

        if prob < 0.20:
            st.success("**Low Risk**  \nModel thinks this loan has good odds of being repaid.")
        elif prob < 0.50:
            st.warning("**Medium Risk**  \nCaution advised — some concerning factors detected.")
        else:
            st.error("**High Risk**  \nStrong warning — multiple important risk signals present.")

    with right:
        st.subheader("Why this score? (SHAP Waterfall Explanation)")
        shap_vals = explainer.shap_values(input_df)
        fig = plt.figure(figsize=(10, 7))
        shap.plots.waterfall(shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=expected_columns
        ), show=False)
        st.pyplot(fig)


with st.expander("📖 How to read the explanation (SHAP waterfall)", expanded=False):
    st.markdown("""
    This chart shows **why** the model gave this risk score:

    - The gray line in the middle (E[f(x)]) is the **average prediction** (~1-2% default rate).
    - Each row = one factor about the applicant.
    - **Red bars to the right** = this factor **increased** the risk score
    - **Blue bars to the left** = this factor **decreased** the risk score
    - Longer bar = bigger influence on this specific decision

    **Examples of what you might see:**
    - High debt-to-income ratio → big red bar → pushes risk up
    - Very good credit history length → blue bar → protects / lowers risk
    - Loan grade D or worse → usually the strongest risk driver
    """)
st.markdown("---")
st.markdown("**Want to test another scenario?**  \nChange any value in the sidebar and click **Calculate Risk Score** again.")
st.button("← Try a different case", type="secondary", use_container_width=True)

st.markdown("---")
st.caption("Model: XGBoost • Explainability: SHAP • Dataset inspiration: standard credit risk benchmark")