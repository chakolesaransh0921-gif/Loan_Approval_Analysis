import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import exp
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ------------------ THEMES ------------------
THEMES = {
    "Ocean": ["#0ea5e9", "#22d3ee", "#3b82f6", "#6366f1", "#a78bfa"],
    "Sunset": ["#f97316", "#fb7185", "#f59e0b", "#fde047", "#ef4444"],
    "Forest": ["#10b981", "#34d399", "#22c55e", "#84cc16", "#16a34a"],
    "Monochrome": ["#111827", "#1f2937", "#374151", "#6b7280", "#9ca3af"],
    "Nebula": ["#f91b6a", "#fa5292", "#fdcb6e", "#fde683", "#f0f8ff"]
}

# ------------------ Helpers ------------------
def emi_from_principal(P, annual_rate_pct, months):
    r = (annual_rate_pct / 100.0) / 12.0
    if months <= 0:
        return 0.0
    if r == 0:
        return P / months
    return P * r * ((1 + r) ** months) / (((1 + r) ** months) - 1)

def principal_from_emi(emi, annual_rate_pct, months):
    r = (annual_rate_pct / 100.0) / 12.0
    if months <= 0:
        return 0.0
    if r == 0:
        return emi * months
    return emi * (((1 + r) ** months) - 1) / (r * ((1 + r) ** months))

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("LP_Train.csv")
    df.rename(columns={
        "ApplicantIncome": "Applicant_Income",
        "CoapplicantIncome": "Coapplicant_Income",
        "LoanAmount": "Loan_Amount"
    }, inplace=True)

    df["Gender"] = df["Gender"].fillna("Male")
    df["Married"] = df["Married"].fillna("Yes")
    df["Dependents"] = df["Dependents"].fillna(0)
    df["Self_Employed"] = df["Self_Employed"].fillna("No")
    df["Loan_Amount"] = df["Loan_Amount"].fillna(df["Loan_Amount"].mean())
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mean())
    df["Credit_History"] = df["Credit_History"].fillna(1.0)

    df["Dependents"] = df["Dependents"].astype(int).astype(str).str.replace("+", "").astype(int)
    df["Total_Income"] = df["Applicant_Income"] + df["Coapplicant_Income"]
    return df

df = load_data()

# ------------------ SIDEBAR: SETTINGS ------------------
st.sidebar.title("⚙️ Loan Approval Dashboard Settings")

with st.sidebar.expander("🎨 Theme & Colors"):
    theme = st.sidebar.selectbox("Select Theme", list(THEMES.keys()))
    palette = THEMES[theme]
    st.sidebar.markdown(f"Current palette: {', '.join(palette)}")
    sns.set_theme(style="whitegrid")
    sns.set_palette(palette)

with st.sidebar.expander("📊 Financial Parameters"):
    interest_rate = st.sidebar.slider("Annual Interest Rate (%)", 6.0, 18.0, 10.0, 0.1)
    max_dti_pct = st.sidebar.slider("Maximum DTI (%)", 10, 60, 40)
    max_dti = max_dti_pct / 100.0

with st.sidebar.expander("👤 Applicant Info"):
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    married = st.sidebar.selectbox("Married", ["Yes", "No"])
    education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.sidebar.selectbox("Self-Employed", ["Yes", "No"])
    dependents = st.sidebar.slider("Dependents", 0, 5, 0)
    property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with st.sidebar.expander("💰 Financial Inputs"):
    app_income = st.sidebar.number_input("Applicant Income (₹)", 0, 1_000_000, 50000, step=500)
    co_income = st.sidebar.number_input("Coapplicant Income (₹)", 0, 1_000_000, 0, step=500)
    loan_amt = st.sidebar.number_input("Loan Amount (₹ thousands)", 0, 10_000, 150, step=10)
    loan_term = st.sidebar.slider("Loan Term (Months)", 120, 480, 360)
    credit_hist = st.sidebar.selectbox("Credit History", [1.0, 0.0])

# ------------------ MAIN PREDICTION LOGIC ------------------
def calculate_loan_eligibility():
    principal = loan_amt * 1000
    total_income = app_income + co_income

    emi = emi_from_principal(principal, interest_rate, loan_term)
    dti = emi / total_income if total_income > 0 else 1.0

    score = 0
    score += 3 if credit_hist == 1.0 else -1
    score += 2 if total_income > 80000 else (1 if total_income > 40000 else 0)
    score += 2 if loan_amt < 200 else (1 if loan_amt < 300 else 0)
    score += 0.5 if education == "Graduate" else 0
    score += 0.5 if property_area != "Rural" else 0
    score += 2 if dti <= max_dti else (-2 if dti > max_dti + 0.1 else 1)
    score -= 0.5 if dependents >= 3 else 0

    approval_prob = 1 / (1 + np.exp(-1.1 * (score - 4.5)))
    approval_prob = clamp(approval_prob, 0.03, 0.97)

    safe_principal = principal_from_emi((total_income * max_dti), interest_rate, loan_term)
    safe_k = safe_principal / 1000

    return approval_prob, emi, safe_k

prob, emi, safe_k = calculate_loan_eligibility()

# ------------------ MAIN DASHBOARD ------------------
st.title("🏦 Loan Approval Analysis Dashboard")
st.markdown("A simple, color-rich tool to predict loan approval, visualize data, and explore what-if scenarios.")

tab1, tab2, tab3, tab4 = st.tabs([
    "Quick Prediction",
    "Advanced Inputs",
    "Bulk CSV Check",
    "Data Explorer"
])

with tab1:
    st.subheader("📊 Quick Prediction Results")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Approval Probability", f"{int(prob * 100)}%", delta=" {:.1f}%".format(prob * 100))
        st.progress(prob)

    with col2:
        st.metric("Estimated EMI (₹)", f"{int(emi):,}", delta=" {:.1f} ₹/month".format(emi))
        st.metric("Safe Loan Amount (₹ thousands)", f"{int(safe_k):,}", delta=" {:.1f}k".format(safe_k))

    with col3:
        st.success("Approved ✅" if prob >= 0.5 else "Not Approved ❌")
        st.info("Note: This is a simple rule-based demo. For real decisions, use a trained model.")

    st.markdown("### 🔍 How It Works")
    st.markdown("""
    - **Score**: Based on credit history, income, loan amount, DTI, education, property area, dependents.
    - **Probability**: Score mapped to 0–100% via logistic function.
    - **Safe Loan**: Calculated as EMI ≤ max DTI% of income.
    """)

with tab2:
    st.subheader("🎛️ Advanced Inputs")
    st.markdown("Adjust parameters, see live updates, and explore different scenarios.")

    st.markdown("#### 🏦 Financial Parameters")
    interest_rate = st.slider("Annual Interest Rate (%)", 6.0, 18.0, 10.0, 0.1, key="advanced_interest")
    max_dti_pct = st.slider("Maximum DTI (%)", 10, 60, 40, key="advanced_dti")
    max_dti = max_dti_pct / 100.0

    st.markdown("#### 👤 Applicant Info")
    gender = st.selectbox("Gender", ["Male", "Female"], key="advanced_gender")
    married = st.selectbox("Married", ["Yes", "No"], key="advanced_married")
    education = st.selectbox("Education", ["Graduate", "Not Graduate"], key="advanced_education")
    self_employed = st.selectbox("Self-Employed", ["Yes", "No"], key="advanced_self")
    dependents = st.slider("Dependents", 0, 5, 0, key="advanced_dep")
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], key="advanced_area")

    st.markdown("#### 💰 Financial Inputs")
    app_income = st.number_input("Applicant Income (₹)", 0, 1_000_000, 50000, step=500, key="advanced_app_income")
    co_income = st.number_input("Coapplicant Income (₹)", 0, 1_000_000, 0, step=500, key="advanced_co_income")
    loan_amt = st.number_input("Loan Amount (₹ thousands)", 0, 10_000, 150, step=10, key="advanced_loan_amt")
    loan_term = st.slider("Loan Term (Months)", 120, 480, 360, key="advanced_term")
    credit_hist = st.selectbox("Credit History", [1.0, 0.0], key="advanced_credit_hist")

    prob, emi, safe_k = calculate_loan_eligibility()
    st.metric("Approval Probability", f"{int(prob * 100)}%")
    st.metric("Estimated EMI (₹)", f"{int(emi):,}")
    st.metric("Safe Loan Amount (₹ thousands)", f"{int(safe_k):,}")
    st.success("Approved ✅" if prob >= 0.5 else "Not Approved ❌")

with tab3:
    st.subheader("📂 Bulk CSV Check")
    st.markdown("Upload your CSV file (same columns as the sample) to check multiple applicants at once.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="bulk_upload")

    if uploaded_file is not None:
        try:
            batch = pd.read_csv(uploaded_file)
            batch = batch.rename(columns={
                "ApplicantIncome": "Applicant_Income",
                "CoapplicantIncome": "Coapplicant_Income",
                "LoanAmount": "Loan_Amount"
            })

            # Fill missing values
            batch["Gender"] = batch["Gender"].fillna("Male")
            batch["Married"] = batch["Married"].fillna("Yes")
            batch["Dependents"] = batch["Dependents"].fillna(0)
            batch["Self_Employed"] = batch["Self_Employed"].fillna("No")
            batch["Loan_Amount"] = batch["Loan_Amount"].fillna(batch["Loan_Amount"].mean())
            batch["Loan_Amount_Term"] = batch["Loan_Amount_Term"].fillna(batch["Loan_Amount_Term"].mean())
            batch["Credit_History"] = batch["Credit_History"].fillna(1.0)

            batch["Dependents"] = batch["Dependents"].astype(int).astype(str).str.replace("+", "").astype(int)
            batch["Total_Income"] = batch["Applicant_Income"] + batch["Coapplicant_Income"]

            results = []
            for _, r in batch.iterrows():
                principal = r["Loan_Amount"] * 1000
                total_income = r["Applicant_Income"] + r["Coapplicant_Income"]
                emi = emi_from_principal(principal, interest_rate, r["Loan_Amount_Term"])
                dti = emi / total_income if total_income > 0 else 1.0

                score = 0
                score += 3 if r["Credit_History"] == 1.0 else -1
                score += 2 if total_income > 80000 else (1 if total_income > 40000 else 0)
                score += 2 if r["Loan_Amount"] < 200 else (1 if r["Loan_Amount"] < 300 else 0)
                score += 0.5 if r["Education"] == "Graduate" else 0
                score += 0.5 if r["Property_Area"] != "Rural" else 0
                score += 2 if dti <= max_dti else (-2 if dti > max_dti + 0.1 else 1)
                score -= 0.5 if r["Dependents"] >= 3 else 0

                approval_prob = 1 / (1 + np.exp(-1.1 * (score - 4.5)))
                approval_prob = clamp(approval_prob, 0.03, 0.97)

                safe_principal = principal_from_emi((total_income * max_dti), interest_rate, r["Loan_Amount_Term"])
                safe_k = safe_principal / 1000

                results.append({
                    "Approval Probability (%)": f"{int(approval_prob*100)}%",
                    "Estimated EMI (₹)": f"{int(emi):,}",
                    "Safe Loan Amount (₹ thousands)": f"{int(safe_k):,}",
                    "Decision": "Approved ✅" if approval_prob >= 0.5 else "Not Approved ❌"
                })

            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

            if st.button("Download Results"):
                st.download_button(
                    label="Download Results (CSV)",
                    data=results_df.to_csv(index=False).encode('utf-8'),
                    file_name="loan_results.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")

with tab4:
    st.subheader("📊 Data Explorer")
    st.markdown("Visualize the dataset and explore correlations.")

    st.markdown("### 📊 Loan Status Distribution")
    fig, ax = plt.subplots(figsize=(5,4))
    sns.countplot(x=df["Loan_Status"], palette=palette, ax=ax)
    ax.set_title("Loan Status Distribution")
    st.pyplot(fig)

    st.markdown("### 📊 Income vs Loan Status (Box Plot)")
    fig, ax = plt.subplots(figsize=(5,4))
    sns.boxplot(x=df["Loan_Status"], y=df["Applicant_Income"], palette=palette, ax=ax)
    ax.set_title("Applicant Income vs Loan Status")
    st.pyplot(fig)

    st.markdown("### 📊 Coapplicant Income vs Loan Status (Box Plot)")
    fig, ax = plt.subplots(figsize=(5,4))
    sns.boxplot(x=df["Loan_Status"], y=df["Coapplicant_Income"], palette=palette, ax=ax)
    ax.set_title("Coapplicant Income vs Loan Status")
    st.pyplot(fig)

    st.markdown("### 📊 Correlation Matrix (Key Numerical Features)")
    st.dataframe(df[['Applicant_Income', 'Coapplicant_Income', 'Loan_Amount']].corr())

    st.markdown("### 🌐 3D Scatter: Income vs Loan Amount vs Status")
    if enable_3d:
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': '3d'}]])
        fig.update_layout(title="3D: Applicant Income vs Coapplicant Income vs Loan Amount")

        fig.add_trace(go.Scatter3d(
            x=df['Applicant_Income'],
            y=df['Coapplicant_Income'],
            z=df['Loan_Amount'],
            mode='markers',
            marker=dict(
                size=5,
                color=df['Loan_Status'].map({'Y': '#4ECDC4', 'N': '#FF6B6B'}),
                opacity=0.7
            ),
            name="Loan Status"
        ))

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("3D visualization disabled. Plotly is not installed.")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("© 2024 Loan Approval Analysis Demo. Simple rule-based predictor. Not a bank decision system.")
