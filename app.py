import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from math import exp

# Optional Plotly (3D)
try:
    import plotly.express as px
    PLOTLY_OK = True
except:
    PLOTLY_OK = False

st.set_page_config(page_title="Loan Approval App", layout="wide")

# ---------------- LOAD DATA ----------------
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
    df["Self_Employed"] = df["Self_Employed"].fillna("No")
    df["Loan_Amount"] = df["Loan_Amount"].fillna(df["Loan_Amount"].mean())
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mean())
    df["Credit_History"] = df["Credit_History"].fillna(1.0)

    # ✅ FIXED DEPENDENTS
    df["Dependents"] = (
        df["Dependents"]
        .astype(str)
        .str.replace("+", "", regex=False)
        .replace("nan", "0")
        .astype(int)
    )

    df["Total_Income"] = df["Applicant_Income"] + df["Coapplicant_Income"]
    return df

df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.header("Applicant Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_emp = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
dependents = st.sidebar.slider("Dependents", 0, 5, 0)
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

app_income = st.sidebar.number_input("Applicant Income", 0, 1_000_000, 5000)
co_income = st.sidebar.number_input("Coapplicant Income", 0, 1_000_000, 0)
loan_amt = st.sidebar.number_input("Loan Amount (in thousands)", 0, 10000, 150)
loan_term = st.sidebar.slider("Loan Term (months)", 120, 480, 360)
credit_hist = st.sidebar.selectbox("Credit History", [1.0, 0.0])

interest_rate = st.sidebar.slider("Interest Rate (%)", 6.0, 18.0, 10.0)

# ✅ FIXED
enable_3d = st.sidebar.checkbox("Enable 3D Visualization", True)

# ---------------- EMI FUNCTIONS ----------------
def emi(P, rate, months):
    r = (rate / 100) / 12
    if r == 0:
        return P / months
    return P * r * ((1 + r) ** months) / (((1 + r) ** months) - 1)

# ---------------- PREDICTION ----------------
def approval_probability():
    principal = loan_amt * 1000
    total_income = app_income + co_income

    e = emi(principal, interest_rate, loan_term)
    dti = e / total_income if total_income > 0 else 1

    score = 0
    score += 3 if credit_hist == 1 else -2
    score += 2 if total_income > 8000 else 1
    score += 2 if loan_amt < 250 else 0
    score += 1 if education == "Graduate" else 0
    score += 1 if property_area != "Rural" else 0
    score -= 1 if dependents >= 3 else 0
    score -= 2 if dti > 0.4 else 1

    prob = 1 / (1 + exp(-(score - 4)))
    return prob, e

prob, emi_val = approval_probability()

# ---------------- MAIN UI ----------------
st.title("🏦 Loan Approval Prediction")

col1, col2, col3 = st.columns(3)
col1.metric("Approval Probability", f"{int(prob*100)}%")
col2.metric("Estimated EMI", f"₹ {int(emi_val):,}")
col3.metric("Status", "Approved ✅" if prob >= 0.5 else "Rejected ❌")

st.progress(int(prob * 100))

# ---------------- DATA VISUALS ----------------
st.subheader("Loan Status Distribution")
fig, ax = plt.subplots()
sb.countplot(x=df["Loan_Status"], ax=ax)
st.pyplot(fig)

st.subheader("Income vs Loan Status")
fig2, ax2 = plt.subplots()
sb.boxplot(x=df["Loan_Status"], y=df["Applicant_Income"], ax=ax2)
st.pyplot(fig2)

if enable_3d and PLOTLY_OK:
    st.subheader("3D Income vs Loan")
    fig3 = px.scatter_3d(
        df,
        x="Applicant_Income",
        y="Coapplicant_Income",
        z="Loan_Amount",
        color="Loan_Status"
    )
    st.plotly_chart(fig3, use_container_width=True)
