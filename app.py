import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from math import exp

# Optional: Plotly for 3D
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

st.set_page_config(page_title="Loan Approval Analysis", layout="wide")

# ------------------ Theming ------------------
THEMES = {
    "Ocean": ["#0ea5e9", "#22d3ee", "#3b82f6", "#6366f1", "#a78bfa"],
    "Sunset": ["#f97316", "#fb7185", "#f59e0b", "#fde047", "#ef4444"],
    "Forest": ["#10b981", "#34d399", "#22c55e", "#84cc16", "#16a34a"],
    "Monochrome": ["#111827", "#1f2937", "#374151", "#6b7280", "#9ca3af"]
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

# ------------------ Load & Clean Data ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("LP_Train.csv")

    df = df.rename(columns={
        "ApplicantIncome": "Applicant_Income",
        "CoapplicantIncome": "Coapplicant_Income",
        "LoanAmount": "Loan_Amount"
    })

    df["Gender"] = df["Gender"].fillna("Male")
    df["Married"] = df["Married"].fillna("Yes")
    df["Dependents"] = df["Dependents"].fillna(0)
    df["Self_Employed"] = df["Self_Employed"].fillna("No")
    df["Loan_Amount"] = df["Loan_Amount"].fillna(df["Loan_Amount"].mean())
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mean())
    df["Credit_History"] = df["Credit_History"].fillna(1.0)

    df["Dependents"] = (
        df["Dependents"]
        .astype(str)
        .str.replace("[+]", "", regex=True)
        .astype(int)
    )

    df["Total_Income"] = df["Applicant_Income"] + df["Coapplicant_Income"]
    return df

df = load_data()

# ------------------ Sidebar: Theme & Global Settings ------------------
st.sidebar.header("Settings")

theme_name = st.sidebar.selectbox("Color Theme", list(THEMES.keys()))
palette = THEMES[theme_name]

sb.set_theme(style="whitegrid")
sb.set_palette(palette)

enable_3d = st.sidebar.checkbox("Enable 3D Exploration", True)
if enable_3d and not PLOTLY_OK:
    st.sidebar.warning("Plotly not installed; 3D disabled.")
    enable_3d = False

interest_rate = st.sidebar.slider("Annual Interest Rate (%)", 6.0, 18.0, 10.0, 0.1)
max_dti_pct = st.sidebar.slider("Max EMI as % of Income", 10, 60, 40)
max_dti = max_dti_pct / 100

# ------------------ Sidebar: Applicant Inputs ------------------
st.sidebar.header("Applicant Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_emp = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
dependents = st.sidebar.slider("Dependents", 0, 5, 0)
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

app_income = st.sidebar.number_input("Applicant Income (monthly)", 0, 1_000_000, 5000, step=500)
co_income = st.sidebar.number_input("Coapplicant Income (monthly)", 0, 1_000_000, 0, step=500)

loan_amt = st.sidebar.number_input("Loan Amount (in thousands)", 0, 10_000, 150, step=10)
loan_term = st.sidebar.slider("Loan Term (months)", 120, 480, 360)
credit_hist = st.sidebar.selectbox("Credit History", [1.0, 0.0])

# ------------------ Rule-Based Prediction ------------------
def approval_probability():
    principal = loan_amt * 1000
    total_income = app_income + co_income

    emi = emi_from_principal(principal, interest_rate, loan_term)
    dti = emi / total_income if total_income > 0 else 1

    score = 0
    score += 3 if credit_hist == 1 else -1
    score += 2 if total_income > 8000 else (1 if total_income > 4000 else 0)
    score += 2 if loan_amt < 200 else (1 if loan_amt < 300 else 0)
    score += 0.5 if education == "Graduate" else 0
    score += 0.5 if property_area != "Rural" else 0
    score += 2 if dti <= max_dti else (-2 if dti > max_dti + 0.1 else 1)
    score -= 0.5 if dependents >= 3 else 0

    p = 1 / (1 + exp(-1.1 * (score - 4.5)))
    p = clamp(p, 0.03, 0.97)

    emi_cap = total_income * max_dti
    safe_principal = principal_from_emi(emi_cap, interest_rate, loan_term)

    return p, emi, safe_principal / 1000

prob, emi, safe_k = approval_probability()

# ------------------ Main Dashboard ------------------
st.title("🏦 Loan Approval Analysis Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "Quick Prediction",
    "What-if Analysis",
    "Bulk Check",
    "Explore Data"
])

with tab1:
    st.metric("Approval Chance", f"{int(prob*100)}%")
    st.progress(int(prob*100))
    st.metric("Estimated EMI", f"{int(emi):,}")
    st.metric("Safe Loan (k)", f"{int(safe_k):,}")
    st.success("Approved ✅" if prob >= 0.5 else "Not Approved ❌")

with tab2:
    st.info("Lower interest or longer tenure increases loan eligibility.")

with tab3:
    st.info("Upload CSV to bulk-check loan eligibility (same column format).")

with tab4:
    st.subheader("Loan Status Distribution")
    fig, ax = plt.subplots()
    sb.countplot(x=df["Loan_Status"], ax=ax, palette=palette[:2])
    st.pyplot(fig)

    st.subheader("Income vs Loan Status")
    fig2, ax2 = plt.subplots()
    sb.boxplot(x=df["Loan_Status"], y=df["Applicant_Income"], ax=ax2)
    st.pyplot(fig2)

    if enable_3d:
        fig3 = px.scatter_3d(
            df,
            x="Applicant_Income",
            y="Coapplicant_Income",
            z="Loan_Amount",
            color="Loan_Status",
            color_discrete_sequence=palette[:2]
        )
        st.plotly_chart(fig3, use_container_width=True)

with st.expander("How it works"):
    st.write("""
    This is a rule-based demo using income, credit history,
    loan size, DTI, and basic demographics.
    Not a real banking model.
    """)
