import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

st.set_page_config(page_title="Loan Approval Analysis", layout="wide")

# ------------------ Load & Clean Data ------------------
@st.cache_data
def load_data():
    df = pd.read_csv('LP_Train.csv')
    df = df.rename(columns={
        'ApplicantIncome': 'Applicant_Income',
        'CoapplicantIncome': 'Coapplicant_Income',
        'LoanAmount': 'Loan_Amount'
    })

    df.Gender = df.Gender.fillna('Male')
    df.Married = df.Married.fillna('Yes')
    df.Dependents = df.Dependents.fillna(0)
    df.Self_Employed = df.Self_Employed.fillna('No')
    df.Loan_Amount = df.Loan_Amount.fillna(df.Loan_Amount.mean())
    df.Loan_Amount_Term = df.Loan_Amount_Term.fillna(df.Loan_Amount_Term.mean())
    df.Credit_History = df.Credit_History.fillna(1.0)

    df.Dependents = df.Dependents.replace('[+]', '', regex=True).astype(int)

    return df


df = load_data()

# ------------------ Sidebar User Input ------------------
st.sidebar.header("Applicant Details")

gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
married = st.sidebar.selectbox("Married", ['Yes', 'No'])
education = st.sidebar.selectbox("Education", ['Graduate', 'Not Graduate'])
self_emp = st.sidebar.selectbox("Self Employed", ['Yes', 'No'])
dependents = st.sidebar.slider("Dependents", 0, 5, 0)
property_area = st.sidebar.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])
app_income = st.sidebar.number_input("Applicant Income", 0, 100000, 5000)
co_income = st.sidebar.number_input("Coapplicant Income", 0, 50000, 0)
loan_amt = st.sidebar.number_input("Loan Amount", 0, 700, 150)
loan_term = st.sidebar.slider("Loan Term (Months)", 120, 480, 360)
credit_hist = st.sidebar.selectbox("Credit History", [1.0, 0.0])

# ------------------ Simple Rule-Based Prediction ------------------
st.sidebar.subheader("Prediction")

def predict_loan():
    score = 0
    if credit_hist == 1.0:
        score += 3
    if app_income > 4000:
        score += 2
    if loan_amt < 300:
        score += 1
    if education == 'Graduate':
        score += 1
    if property_area != 'Rural':
        score += 1

    return "Approved ✅" if score >= 5 else "Not Approved ❌"

result = predict_loan()
st.sidebar.success(f"Loan Status: {result}")

# ------------------ Main Dashboard ------------------
st.title("🏦 Loan Approval Analysis Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Loan Status Distribution")
    fig, ax = plt.subplots()
    sb.countplot(x=df['Loan_Status'], ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Credit History vs Loan Status")
    fig, ax = plt.subplots()
    pd.crosstab(df['Credit_History'], df['Loan_Status']).plot(kind='bar', ax=ax)
    st.pyplot(fig)

st.subheader("Income Analysis")
col3, col4 = st.columns(2)

with col3:
    fig, ax = plt.subplots()
    sb.boxplot(x=df['Loan_Status'], y=df['Applicant_Income'], ax=ax)
    st.pyplot(fig)

with col4:
    fig, ax = plt.subplots()
    sb.barplot(x=df['Loan_Status'], y=df['Coapplicant_Income'], ax=ax)
    st.pyplot(fig)

st.subheader("Property Area Impact")
fig, ax = plt.subplots()
pd.crosstab(df['Property_Area'], df['Loan_Status']).plot(kind='bar', ax=ax)
plt.xticks(rotation=0)
st.pyplot(fig)

st.subheader("Correlation Matrix")
st.dataframe(df[['Applicant_Income', 'Coapplicant_Income', 'Loan_Amount']].corr())

st.markdown("""
### 🔍 How Prediction Works
- Credit history has highest weight
- Higher income improves approval chances
- Lower loan amount = safer approval
- Graduates & urban/semiurban applicants score better

⚠️ This is a **rule-based demo model**, not a real bank decision system.
""")
