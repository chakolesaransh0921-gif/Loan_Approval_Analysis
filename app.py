import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG & GLOBAL STYLE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LoanIQ – Loan Approval Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
      font-family: 'DM Sans', sans-serif;
      background-color: #0a0e1a;
      color: #e2e8f0;
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0f1629 0%, #111827 100%);
      border-right: 1px solid #1e293b;
  }
  section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

  /* ── Top nav tabs ── */
  .stTabs [data-baseweb="tab-list"] {
      gap: 6px;
      background: #111827;
      border-radius: 12px;
      padding: 6px;
      border: 1px solid #1e293b;
  }
  .stTabs [data-baseweb="tab"] {
      background: transparent;
      border-radius: 8px;
      color: #64748b !important;
      font-family: 'Syne', sans-serif;
      font-weight: 700;
      font-size: 0.85rem;
      letter-spacing: .5px;
      padding: 10px 20px;
      border: none !important;
      transition: all .25s ease;
  }
  .stTabs [aria-selected="true"] {
      background: linear-gradient(135deg,#6366f1,#a855f7) !important;
      color: #fff !important;
      box-shadow: 0 4px 20px rgba(99,102,241,.45);
  }
  .stTabs [data-baseweb="tab-panel"] {
      background: transparent;
      padding-top: 20px;
  }

  /* ── Metric cards ── */
  [data-testid="stMetric"] {
      background: linear-gradient(135deg,#111827,#1e293b);
      border: 1px solid #334155;
      border-radius: 16px;
      padding: 20px 24px !important;
      box-shadow: 0 4px 24px rgba(0,0,0,.4);
  }
  [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size:.8rem; letter-spacing:1px; text-transform:uppercase; }
  [data-testid="stMetricValue"] { color: #f8fafc !important; font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; }
  [data-testid="stMetricDelta"] svg { display:none; }

  /* ── Section titles ── */
  .section-title {
      font-family: 'Syne', sans-serif;
      font-size: 1.4rem;
      font-weight: 800;
      letter-spacing: -.5px;
      background: linear-gradient(90deg,#818cf8,#c084fc);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 16px;
  }

  /* ── Card container ── */
  .glass-card {
      background: rgba(17,24,39,.85);
      border: 1px solid #1e293b;
      border-radius: 20px;
      padding: 24px;
      margin-bottom: 18px;
      backdrop-filter: blur(12px);
      box-shadow: 0 8px 32px rgba(0,0,0,.3);
  }

  /* ── Prediction badge ── */
  .pred-approved {
      background: linear-gradient(135deg,#059669,#10b981);
      color: #fff;
      border-radius: 50px;
      padding: 10px 28px;
      font-family: 'Syne',sans-serif;
      font-weight: 800;
      font-size: 1.1rem;
      text-align: center;
      box-shadow: 0 6px 24px rgba(16,185,129,.4);
  }
  .pred-denied {
      background: linear-gradient(135deg,#dc2626,#ef4444);
      color: #fff;
      border-radius: 50px;
      padding: 10px 28px;
      font-family: 'Syne',sans-serif;
      font-weight: 800;
      font-size: 1.1rem;
      text-align: center;
      box-shadow: 0 6px 24px rgba(239,68,68,.4);
  }

  /* ── Progress bars ── */
  .stProgress > div > div { background: linear-gradient(90deg,#6366f1,#a855f7); border-radius:8px; }

  /* ── Buttons ── */
  .stButton > button {
      background: linear-gradient(135deg,#6366f1,#a855f7);
      color: #fff;
      border: none;
      border-radius: 10px;
      font-family: 'Syne',sans-serif;
      font-weight: 700;
      padding: 10px 28px;
      transition: all .2s ease;
      box-shadow: 0 4px 18px rgba(99,102,241,.4);
  }
  .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(99,102,241,.55); }

  /* ── Dataframe ── */
  .stDataFrame { border-radius: 12px; overflow: hidden; }

  /* ── Info boxes ── */
  .stInfo { background: #1e1b4b; border-left: 4px solid #6366f1; border-radius: 8px; }
  .stWarning { background: #292524; border-left: 4px solid #f59e0b; border-radius: 8px; }
  .stSuccess { background: #052e16; border-left: 4px solid #10b981; border-radius: 8px; }

  /* ── Hero ── */
  .hero-title {
      font-family: 'Syne', sans-serif;
      font-size: 3rem;
      font-weight: 800;
      background: linear-gradient(135deg,#818cf8 0%,#c084fc 50%,#f472b6 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      line-height: 1.1;
  }
  .hero-sub {
      color: #64748b;
      font-size: 1.05rem;
      font-weight: 300;
      margin-top: 8px;
  }

  div[data-testid="column"] { gap: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PLOTLY DARK TEMPLATE
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans', color='#94a3b8'),
    title_font=dict(family='Syne', color='#f1f5f9', size=16),
    legend=dict(bgcolor='rgba(17,24,39,.7)', bordercolor='#334155', borderwidth=1),
    xaxis=dict(gridcolor='#1e293b', linecolor='#334155', tickfont=dict(color='#64748b')),
    yaxis=dict(gridcolor='#1e293b', linecolor='#334155', tickfont=dict(color='#64748b')),
    margin=dict(l=10, r=10, t=50, b=10),
)
PALETTE = ['#6366f1','#a855f7','#ec4899','#f59e0b','#10b981','#06b6d4','#f97316']


# ─────────────────────────────────────────────
#  DATA — synthetic fallback if CSV missing
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('LP_Train.csv')
    except FileNotFoundError:
        np.random.seed(42)
        n = 614
        df = pd.DataFrame({
            'Loan_ID': [f'LP{str(i).zfill(6)}' for i in range(n)],
            'Gender': np.random.choice(['Male','Female'], n, p=[.81,.19]),
            'Married': np.random.choice(['Yes','No'], n, p=[.65,.35]),
            'Dependents': np.random.choice(['0','1','2','3+'], n, p=[.57,.17,.16,.10]),
            'Education': np.random.choice(['Graduate','Not Graduate'], n, p=[.78,.22]),
            'Self_Employed': np.random.choice(['Yes','No'], n, p=[.14,.86]),
            'ApplicantIncome': np.random.lognormal(8.5,.6,n).astype(int),
            'CoapplicantIncome': np.where(np.random.random(n)>.4, np.random.lognormal(7.5,.7,n).astype(int), 0),
            'LoanAmount': np.random.lognormal(4.9,.5,n),
            'Loan_Amount_Term': np.random.choice([120,180,240,300,360,480], n, p=[.03,.04,.05,.05,.80,.03]),
            'Credit_History': np.random.choice([1.0,0.0], n, p=[.84,.16]),
            'Property_Area': np.random.choice(['Urban','Semiurban','Rural'], n, p=[.38,.32,.30]),
            'Loan_Status': np.random.choice(['Y','N'], n, p=[.69,.31]),
        })

    df = df.rename(columns={
        'ApplicantIncome': 'Applicant_Income',
        'CoapplicantIncome': 'Coapplicant_Income',
        'LoanAmount': 'Loan_Amount'
    })

    df['Gender']        = df['Gender'].fillna('Male')
    df['Married']       = df['Married'].fillna('Yes')
    df['Dependents']    = df['Dependents'].fillna(0)
    df['Self_Employed'] = df['Self_Employed'].fillna('No')
    df['Loan_Amount']   = df['Loan_Amount'].fillna(df['Loan_Amount'].mean())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(360)
    df['Credit_History']   = df['Credit_History'].fillna(1.0)
    df['Dependents'] = df['Dependents'].astype(str).str.replace(r'[+]','',regex=True).astype(int)
    df['Total_Income'] = df['Applicant_Income'] + df['Coapplicant_Income']
    df['EMI']          = (df['Loan_Amount'] * 1000) / df['Loan_Amount_Term']
    df['Income_to_Loan'] = df['Total_Income'] / (df['Loan_Amount'] + 1)
    return df

df = load_data()
total = len(df)
approved = (df['Loan_Status']=='Y').sum()
denied    = (df['Loan_Status']=='N').sum()
approval_rate = round(approved/total*100,1)


# ─────────────────────────────────────────────
#  SIDEBAR – APPLICANT INPUT
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-family:Syne;font-size:1.3rem;font-weight:800;color:#818cf8;margin-bottom:4px'>🏦 LoanIQ</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#475569;font-size:.8rem;margin-bottom:20px'>Loan Approval Intelligence</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📋 Applicant Profile")

    gender      = st.selectbox("Gender",          ['Male','Female'])
    married     = st.selectbox("Marital Status",  ['Yes','No'])
    education   = st.selectbox("Education",       ['Graduate','Not Graduate'])
    self_emp    = st.selectbox("Self Employed",   ['No','Yes'])
    dependents  = st.slider("Dependents",          0, 4, 0)
    property_area = st.selectbox("Property Area", ['Urban','Semiurban','Rural'])

    st.markdown("### 💰 Financials")
    app_income  = st.number_input("Applicant Income (₹)",   0, 200000, 5000, step=500)
    co_income   = st.number_input("Coapplicant Income (₹)", 0,  100000, 0, step=500)
    loan_amt    = st.number_input("Loan Amount (₹ 000s)",   0, 700, 150, step=10)
    loan_term   = st.select_slider("Loan Term (months)",    options=[120,180,240,300,360,480], value=360)
    credit_hist = st.selectbox("Credit History",            [1.0, 0.0],
                               format_func=lambda x: "Good (1.0)" if x==1.0 else "Bad (0.0)")

    # ── Score engine ──
    def compute_score():
        score = 0
        factors = {}
        if credit_hist==1.0:  score+=30; factors['Credit History']=30
        else:                            factors['Credit History']=0
        inc_pts = min(20, int(app_income/1000))
        score += inc_pts; factors['Applicant Income'] = inc_pts
        co_pts = min(10, int(co_income/1000))
        score += co_pts; factors['Coapplicant Income'] = co_pts
        loan_pts = max(0, 15 - int(loan_amt/20))
        score += loan_pts; factors['Loan Amount'] = loan_pts
        if education=='Graduate':  score+=10; factors['Education']=10
        else:                                  factors['Education']=0
        if property_area=='Urban':      score+=8; factors['Property Area']=8
        elif property_area=='Semiurban':score+=6; factors['Property Area']=6
        else:                           score+=2; factors['Property Area']=2
        if married=='Yes': score+=5; factors['Married']=5
        else:                          factors['Married']=0
        if dependents==0:  score+=3; factors['Dependents']=3
        else:                          factors['Dependents']=max(0,3-dependents)
        return min(score,100), factors

    score, factors = compute_score()
    emi = (loan_amt*1000)/loan_term if loan_term else 0
    total_inc = app_income + co_income
    dti = (emi/total_inc*100) if total_inc>0 else 100

    st.markdown("---")
    st.markdown("### 🎯 Risk Score")
    st.progress(score/100)
    color = "#10b981" if score>=55 else "#f59e0b" if score>=35 else "#ef4444"
    st.markdown(f"<div style='text-align:center;font-family:Syne;font-size:2rem;font-weight:800;color:{color}'>{score}/100</div>", unsafe_allow_html=True)

    verdict = "Approved ✅" if score>=55 else "Not Approved ❌"
    css_cls = "pred-approved" if score>=55 else "pred-denied"
    st.markdown(f"<div class='{css_cls}' style='margin-top:10px'>{verdict}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;color:#475569;font-size:.75rem;margin-top:6px'>EMI ≈ ₹{emi:,.0f}/mo &nbsp;|&nbsp; DTI {dti:.1f}%</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class='glass-card' style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:16px'>
  <div>
    <div class='hero-title'>Loan Approval Intelligence</div>
    <div class='hero-sub'>Multi-dimensional analytics · Real-time scoring · Risk profiling</div>
  </div>
  <div style='display:flex;gap:12px;flex-wrap:wrap'>
    <div style='background:#1e293b;border:1px solid #334155;border-radius:12px;padding:10px 20px;text-align:center'>
      <div style='color:#818cf8;font-family:Syne;font-size:1.5rem;font-weight:800'>{:,}</div>
      <div style='color:#64748b;font-size:.75rem'>Total Records</div>
    </div>
    <div style='background:#1e293b;border:1px solid #334155;border-radius:12px;padding:10px 20px;text-align:center'>
      <div style='color:#10b981;font-family:Syne;font-size:1.5rem;font-weight:800'>{}</div>
      <div style='color:#64748b;font-size:.75rem'>Approval Rate</div>
    </div>
    <div style='background:#1e293b;border:1px solid #334155;border-radius:12px;padding:10px 20px;text-align:center'>
      <div style='color:#f59e0b;font-family:Syne;font-size:1.5rem;font-weight:800'>{:,}</div>
      <div style='color:#64748b;font-size:.75rem'>Avg Income ₹</div>
    </div>
  </div>
</div>
""".format(total, f"{approval_rate}%", int(df['Applicant_Income'].mean())), unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "👥 Demographics",
    "💰 Income & Loan",
    "📈 Risk Analysis",
    "🤖 Prediction Studio",
    "🗃️ Data Explorer"
])


# ══════════════════════════════════════════════
#  TAB 1 – OVERVIEW
# ══════════════════════════════════════════════
with tabs[0]:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Applications", f"{total:,}")
    c2.metric("Approved",           f"{approved:,}", f"▲ {approval_rate}%")
    c3.metric("Denied",             f"{denied:,}",   f"▼ {100-approval_rate}%")
    c4.metric("Avg Loan Amt",       f"₹{df['Loan_Amount'].mean():.0f}K")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    # Donut – Loan Status
    with col_a:
        st.markdown("<div class='section-title'>Loan Status Distribution</div>", unsafe_allow_html=True)
        counts = df['Loan_Status'].value_counts()
        fig = go.Figure(go.Pie(
            labels=['Approved','Denied'], values=[counts.get('Y',0),counts.get('N',0)],
            hole=.55, marker_colors=['#6366f1','#ec4899'],
            textfont=dict(family='DM Sans',size=14),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>'
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=320,
            annotations=[dict(text=f"<b>{approval_rate}%</b><br>Approved",
                              x=.5,y=.5,font_size=16,showarrow=False,
                              font=dict(color='#f1f5f9',family='Syne'))])
        st.plotly_chart(fig, use_container_width=True)

    # Bar – Property Area
    with col_b:
        st.markdown("<div class='section-title'>Approvals by Property Area</div>", unsafe_allow_html=True)
        pa = pd.crosstab(df['Property_Area'], df['Loan_Status']).reset_index()
        fig = go.Figure()
        fig.add_bar(name='Approved', x=pa['Property_Area'], y=pa.get('Y',pa.iloc[:,1]),
                    marker_color='#6366f1', marker_line_width=0)
        fig.add_bar(name='Denied',   x=pa['Property_Area'], y=pa.get('N',pa.iloc[:,2]),
                    marker_color='#ec4899', marker_line_width=0)
        fig.update_layout(**PLOTLY_LAYOUT, height=320, barmode='group',
                          xaxis_title='Area', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    # Credit History Stacked
    with col_c:
        st.markdown("<div class='section-title'>Credit History vs Loan Status</div>", unsafe_allow_html=True)
        ch = pd.crosstab(df['Credit_History'], df['Loan_Status'], normalize='index')*100
        fig = go.Figure()
        fig.add_bar(name='Approved', x=['Bad Credit','Good Credit'], y=ch.get('Y',[0,0]),
                    marker_color='#10b981', marker_line_width=0)
        fig.add_bar(name='Denied',   x=['Bad Credit','Good Credit'], y=ch.get('N',[0,0]),
                    marker_color='#f43f5e', marker_line_width=0)
        fig.update_layout(**PLOTLY_LAYOUT, height=300, barmode='stack',
                          yaxis_title='% of Applications')
        st.plotly_chart(fig, use_container_width=True)

    # Education pie
    with col_d:
        st.markdown("<div class='section-title'>Education Split</div>", unsafe_allow_html=True)
        ed = df['Education'].value_counts()
        fig = go.Figure(go.Pie(
            labels=ed.index, values=ed.values,
            marker_colors=['#a855f7','#f97316'], hole=.4,
            textfont=dict(family='DM Sans',size=13),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Loan Term distribution
    st.markdown("<div class='section-title'>Loan Term Distribution</div>", unsafe_allow_html=True)
    lt = df['Loan_Amount_Term'].value_counts().sort_index().reset_index()
    lt.columns = ['Term','Count']
    fig = px.bar(lt, x='Term', y='Count',
                 color='Count', color_continuous_scale='Plasma',
                 labels={'Term':'Months','Count':'Applications'})
    fig.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False)
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 2 – DEMOGRAPHICS
# ══════════════════════════════════════════════
with tabs[1]:
    st.markdown("<div class='section-title'>Demographic Deep-Dive</div>", unsafe_allow_html=True)

    r1c1, r1c2, r1c3 = st.columns(3)

    # Gender donut
    with r1c1:
        g = df['Gender'].value_counts()
        fig = go.Figure(go.Pie(labels=g.index, values=g.values,
            hole=.5, marker_colors=['#6366f1','#ec4899'],
            textfont=dict(family='DM Sans',size=12)))
        fig.update_layout(**PLOTLY_LAYOUT, height=260, title='Gender Split')
        st.plotly_chart(fig, use_container_width=True)

    # Married donut
    with r1c2:
        m = df['Married'].value_counts()
        fig = go.Figure(go.Pie(labels=m.index, values=m.values,
            hole=.5, marker_colors=['#a855f7','#f59e0b'],
            textfont=dict(family='DM Sans',size=12)))
        fig.update_layout(**PLOTLY_LAYOUT, height=260, title='Marital Status')
        st.plotly_chart(fig, use_container_width=True)

    # Self Employed donut
    with r1c3:
        se = df['Self_Employed'].value_counts()
        fig = go.Figure(go.Pie(labels=se.index, values=se.values,
            hole=.5, marker_colors=['#10b981','#f43f5e'],
            textfont=dict(family='DM Sans',size=12)))
        fig.update_layout(**PLOTLY_LAYOUT, height=260, title='Employment Type')
        st.plotly_chart(fig, use_container_width=True)

    # Dependents grouped bar
    st.markdown("<div class='section-title'>Dependents vs Approval</div>", unsafe_allow_html=True)
    dep = pd.crosstab(df['Dependents'], df['Loan_Status']).reset_index()
    fig = go.Figure()
    fig.add_bar(name='Approved', x=dep['Dependents'].astype(str), y=dep.get('Y',0),
                marker_color='#6366f1', marker_line_width=0)
    fig.add_bar(name='Denied',   x=dep['Dependents'].astype(str), y=dep.get('N',0),
                marker_color='#ec4899', marker_line_width=0)
    fig.update_layout(**PLOTLY_LAYOUT, height=300, barmode='group',
                      xaxis_title='Number of Dependents')
    st.plotly_chart(fig, use_container_width=True)

    # Sunburst: Gender → Married → Status
    st.markdown("<div class='section-title'>Hierarchical Breakdown – Gender → Married → Status</div>", unsafe_allow_html=True)
    fig = px.sunburst(df, path=['Gender','Married','Loan_Status'],
                      color='Loan_Status',
                      color_discrete_map={'Y':'#6366f1','N':'#ec4899'},
                      maxdepth=3)
    fig.update_layout(**PLOTLY_LAYOUT, height=480)
    st.plotly_chart(fig, use_container_width=True)

    # Treemap: Property Area → Education → Status
    st.markdown("<div class='section-title'>Treemap – Area · Education · Status</div>", unsafe_allow_html=True)
    fig = px.treemap(df, path=['Property_Area','Education','Loan_Status'],
                     color='Loan_Status',
                     color_discrete_map={'Y':'#10b981','N':'#f43f5e'})
    fig.update_layout(**PLOTLY_LAYOUT, height=420)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 3 – INCOME & LOAN
# ══════════════════════════════════════════════
with tabs[2]:
    st.markdown("<div class='section-title'>Financial Analysis</div>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)

    # Applicant Income box
    with c1:
        fig = go.Figure()
        for status, color in [('Y','#6366f1'),('N','#ec4899')]:
            sub = df[df['Loan_Status']==status]
            fig.add_trace(go.Box(y=sub['Applicant_Income'], name='Approved' if status=='Y' else 'Denied',
                                 marker_color=color, boxmean=True,
                                 line=dict(width=2)))
        fig.update_layout(**PLOTLY_LAYOUT, height=340, title='Applicant Income Distribution',
                          yaxis_title='Income (₹)')
        st.plotly_chart(fig, use_container_width=True)

    # Loan Amount violin
    with c2:
        fig = go.Figure()
        for status, color, name in [('Y','#a855f7','Approved'),('N','#f97316','Denied')]:
            sub = df[df['Loan_Status']==status]
            fig.add_trace(go.Violin(y=sub['Loan_Amount'], name=name, box_visible=True,
                                    meanline_visible=True, fillcolor=color,
                                    opacity=.7, line_color=color))
        fig.update_layout(**PLOTLY_LAYOUT, height=340, title='Loan Amount Distribution',
                          yaxis_title='Loan Amount (₹000s)')
        st.plotly_chart(fig, use_container_width=True)

    # Scatter: Income vs Loan Amount
    st.markdown("<div class='section-title'>Income vs Loan Amount – Scatter</div>", unsafe_allow_html=True)
    fig = px.scatter(df, x='Applicant_Income', y='Loan_Amount',
                     color='Loan_Status', size='Total_Income',
                     color_discrete_map={'Y':'#6366f1','N':'#f43f5e'},
                     labels={'Loan_Status':'Status'},
                     opacity=.7, hover_data=['Education','Property_Area'])
    fig.update_layout(**PLOTLY_LAYOUT, height=420,
                      xaxis_title='Applicant Income (₹)',
                      yaxis_title='Loan Amount (₹000s)')
    st.plotly_chart(fig, use_container_width=True)

    c3,c4 = st.columns(2)
    # Income histogram by status
    with c3:
        fig = px.histogram(df, x='Total_Income', color='Loan_Status', nbins=40,
                           color_discrete_map={'Y':'#10b981','N':'#f43f5e'},
                           barmode='overlay', opacity=.7,
                           labels={'Total_Income':'Total Household Income'})
        fig.update_layout(**PLOTLY_LAYOUT, height=320, title='Total Income Histogram')
        st.plotly_chart(fig, use_container_width=True)

    # EMI distribution
    with c4:
        fig = px.histogram(df, x='EMI', color='Loan_Status', nbins=40,
                           color_discrete_map={'Y':'#6366f1','N':'#ec4899'},
                           barmode='overlay', opacity=.7)
        fig.update_layout(**PLOTLY_LAYOUT, height=320, title='Monthly EMI Distribution')
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap – avg loan amt
    st.markdown("<div class='section-title'>Avg Loan Amount – Education × Property Area</div>", unsafe_allow_html=True)
    pivot = df.pivot_table(values='Loan_Amount', index='Education',
                           columns='Property_Area', aggfunc='mean')
    fig = px.imshow(pivot, color_continuous_scale='Plasma', text_auto='.0f',
                    labels=dict(color='Avg Loan (₹K)'))
    fig.update_layout(**PLOTLY_LAYOUT, height=280)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 4 – RISK ANALYSIS
# ══════════════════════════════════════════════
with tabs[3]:
    st.markdown("<div class='section-title'>Risk & Correlation Analysis</div>", unsafe_allow_html=True)

    # Correlation heatmap
    num_cols = ['Applicant_Income','Coapplicant_Income','Loan_Amount',
                'Loan_Amount_Term','Credit_History','Total_Income','EMI','Income_to_Loan']
    corr = df[num_cols].corr()
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1)
    fig.update_layout(**PLOTLY_LAYOUT, height=500, title='Full Correlation Matrix')
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    # Risk gauge: approval rate by credit
    with c1:
        st.markdown("<div class='section-title'>Approval Rate by Credit History</div>", unsafe_allow_html=True)
        cr_rates = df.groupby('Credit_History')['Loan_Status'].apply(
            lambda x: (x=='Y').sum()/len(x)*100).reset_index()
        cr_rates.columns=['Credit','Rate']
        cr_rates['Label'] = cr_rates['Credit'].map({1.0:'Good Credit',0.0:'Bad Credit'})
        fig = go.Figure(go.Bar(
            x=cr_rates['Label'], y=cr_rates['Rate'],
            marker_color=['#10b981','#f43f5e'],
            text=cr_rates['Rate'].apply(lambda v: f"{v:.1f}%"),
            textposition='outside', textfont=dict(color='#f1f5f9'),
            marker_line_width=0
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=320, yaxis_title='Approval Rate %',
                          yaxis_range=[0,110])
        st.plotly_chart(fig, use_container_width=True)

    # Income-to-Loan ratio
    with c2:
        st.markdown("<div class='section-title'>Income-to-Loan Ratio by Status</div>", unsafe_allow_html=True)
        fig = go.Figure()
        for status, color, name in [('Y','#6366f1','Approved'),('N','#ec4899','Denied')]:
            sub = df[df['Loan_Status']==status]['Income_to_Loan'].clip(0,200)
            fig.add_trace(go.Histogram(x=sub, name=name, marker_color=color,
                                       opacity=.7, nbinsx=40))
        fig.update_layout(**PLOTLY_LAYOUT, height=320, barmode='overlay',
                          xaxis_title='Income / Loan Ratio')
        st.plotly_chart(fig, use_container_width=True)

    # 3D scatter
    st.markdown("<div class='section-title'>3D Risk Space – Income · Loan · EMI</div>", unsafe_allow_html=True)
    sample = df.sample(min(300, len(df)), random_state=42)
    fig = px.scatter_3d(sample, x='Applicant_Income', y='Loan_Amount', z='EMI',
                        color='Loan_Status', symbol='Education',
                        color_discrete_map={'Y':'#6366f1','N':'#f43f5e'},
                        opacity=.8)
    fig.update_layout(**PLOTLY_LAYOUT, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Parallel coordinates
    st.markdown("<div class='section-title'>Parallel Coordinates – Multi-Feature Risk View</div>", unsafe_allow_html=True)
    pc_df = df.copy()
    pc_df['Status_Num'] = (pc_df['Loan_Status']=='Y').astype(int)
    fig = px.parallel_coordinates(pc_df,
        dimensions=['Applicant_Income','Coapplicant_Income','Loan_Amount',
                    'Credit_History','Status_Num'],
        color='Status_Num',
        color_continuous_scale=px.colors.diverging.Tealrose,
        labels={'Status_Num':'Approved','Applicant_Income':'App. Income',
                'Coapplicant_Income':'Coapplicant','Loan_Amount':'Loan Amt',
                'Credit_History':'Credit'})
    fig.update_layout(**PLOTLY_LAYOUT, height=420)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 5 – PREDICTION STUDIO
# ══════════════════════════════════════════════
with tabs[4]:
    st.markdown("<div class='section-title'>🤖 Prediction Studio</div>", unsafe_allow_html=True)
    st.info("Adjust the applicant profile in the **sidebar** and see live scoring below.")

    c1, c2 = st.columns([1,1])

    # Factor radar
    with c1:
        st.markdown("<div class='section-title'>Factor Breakdown Radar</div>", unsafe_allow_html=True)
        cats = list(factors.keys())
        vals = list(factors.values())
        max_vals = {'Credit History':30,'Applicant Income':20,'Coapplicant Income':10,
                    'Loan Amount':15,'Education':10,'Property Area':8,'Married':5,'Dependents':3}
        pct_vals = [round(factors.get(c,0)/max_vals.get(c,1)*100) for c in cats]
        fig = go.Figure(go.Scatterpolar(
            r=pct_vals+[pct_vals[0]],
            theta=cats+[cats[0]],
            fill='toself',
            fillcolor='rgba(99,102,241,.25)',
            line=dict(color='#6366f1', width=2),
            marker=dict(color='#a855f7', size=6)
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=400,
            polar=dict(
                bgcolor='rgba(17,24,39,.5)',
                radialaxis=dict(visible=True, range=[0,100],
                                gridcolor='#1e293b', color='#475569'),
                angularaxis=dict(gridcolor='#1e293b', color='#94a3b8')))
        st.plotly_chart(fig, use_container_width=True)

    # Score gauge
    with c2:
        st.markdown("<div class='section-title'>Risk Score Gauge</div>", unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            delta={'reference':55,'increasing':{'color':'#10b981'},'decreasing':{'color':'#f43f5e'}},
            gauge={
                'axis':{'range':[0,100],'tickcolor':'#64748b','tickfont':dict(color='#94a3b8')},
                'bar':{'color':'#6366f1','thickness':.3},
                'bgcolor':'rgba(0,0,0,0)',
                'borderwidth':0,
                'steps':[
                    {'range':[0,35],'color':'rgba(239,68,68,.2)'},
                    {'range':[35,55],'color':'rgba(245,158,11,.2)'},
                    {'range':[55,100],'color':'rgba(16,185,129,.2)'},
                ],
                'threshold':{'line':{'color':'#f8fafc','width':4},'thickness':.8,'value':55}
            },
            title={'text':"Risk Score",'font':{'family':'Syne','color':'#94a3b8','size':16}},
            number={'font':{'family':'Syne','color':'#f1f5f9','size':56}}
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=380)
        st.plotly_chart(fig, use_container_width=True)

    # Factor score bars
    st.markdown("<div class='section-title'>Individual Factor Scores</div>", unsafe_allow_html=True)
    factor_df = pd.DataFrame({'Factor':list(factors.keys()),'Score':list(factors.values()),
                              'Max':list(max_vals.values())})
    factor_df['%'] = factor_df['Score']/factor_df['Max']*100
    fig = go.Figure(go.Bar(
        x=factor_df['%'], y=factor_df['Factor'], orientation='h',
        marker=dict(
            color=factor_df['%'],
            colorscale=[[0,'#f43f5e'],[.5,'#f59e0b'],[1,'#10b981']],
            showscale=True,
            colorbar=dict(title='% Max', tickfont=dict(color='#94a3b8'))
        ),
        text=factor_df.apply(lambda r: f"{r['Score']}/{r['Max']}", axis=1),
        textposition='auto', textfont=dict(color='#fff')
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=340, xaxis_title='% of Max Score',
                      xaxis_range=[0,110])
    st.plotly_chart(fig, use_container_width=True)

    # EMI and DTI summary
    st.markdown("<div class='section-title'>📊 Applicant Summary</div>", unsafe_allow_html=True)
    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Applicant Income", f"₹{app_income:,}")
    s2.metric("Monthly EMI",      f"₹{emi:,.0f}")
    s3.metric("Debt-to-Income",   f"{dti:.1f}%",
              delta="Good" if dti<40 else "High",
              delta_color="normal" if dti<40 else "inverse")
    s4.metric("Risk Score",       f"{score}/100",
              delta="Low Risk" if score>=55 else "High Risk",
              delta_color="normal" if score>=55 else "inverse")

    st.markdown("---")
    st.markdown("""
<div class='glass-card'>
  <div style='font-family:Syne;font-size:1rem;font-weight:700;color:#818cf8;margin-bottom:10px'>⚠️ How scoring works</div>
  <ul style='color:#94a3b8;font-size:.88rem;line-height:2'>
    <li><b>Credit History (30 pts)</b> – Highest weight; good history adds 30 pts</li>
    <li><b>Applicant Income (20 pts)</b> – Scales up to ₹20,000; every ₹1K = 1 pt</li>
    <li><b>Coapplicant Income (10 pts)</b> – Up to ₹10,000; every ₹1K = 1 pt</li>
    <li><b>Loan Amount (15 pts)</b> – Lower loan = safer; penalised after ₹300K</li>
    <li><b>Education (10 pts)</b> – Graduate applicants score higher</li>
    <li><b>Property Area (8 pts)</b> – Urban > Semiurban > Rural</li>
    <li><b>Marital Status (5 pts)</b> – Married applicants score higher</li>
    <li><b>Dependents (3 pts)</b> – Fewer dependents = higher score</li>
    <li><b>Threshold: 55/100</b> → Approved</li>
  </ul>
  <div style='color:#475569;font-size:.75rem;margin-top:8px'>⚠️ Rule-based demo — not a real bank decision system.</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB 6 – DATA EXPLORER
# ══════════════════════════════════════════════
with tabs[5]:
    st.markdown("<div class='section-title'>🗃️ Raw Data Explorer</div>", unsafe_allow_html=True)

    # Filters
    fc1,fc2,fc3,fc4 = st.columns(4)
    f_status  = fc1.multiselect("Loan Status", ['Y','N'], default=['Y','N'])
    f_area    = fc2.multiselect("Property Area", df['Property_Area'].unique().tolist(),
                                default=df['Property_Area'].unique().tolist())
    f_edu     = fc3.multiselect("Education", df['Education'].unique().tolist(),
                                default=df['Education'].unique().tolist())
    f_credit  = fc4.multiselect("Credit History", [1.0,0.0], default=[1.0,0.0])

    filtered = df[
        df['Loan_Status'].isin(f_status) &
        df['Property_Area'].isin(f_area) &
        df['Education'].isin(f_edu) &
        df['Credit_History'].isin(f_credit)
    ]

    st.caption(f"Showing {len(filtered):,} of {total:,} records")

    display_cols = ['Loan_ID','Gender','Married','Education','Self_Employed',
                    'Applicant_Income','Coapplicant_Income','Loan_Amount',
                    'Loan_Amount_Term','Credit_History','Property_Area','Loan_Status']
    existing_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[existing_cols].reset_index(drop=True),
        use_container_width=True,
        height=380
    )

    # Summary stats
    st.markdown("<div class='section-title'>Descriptive Statistics</div>", unsafe_allow_html=True)
    st.dataframe(filtered[['Applicant_Income','Coapplicant_Income',
                            'Loan_Amount','EMI','Total_Income']].describe().round(2),
                 use_container_width=True)

    # Pairplot proxy – scatter matrix
    st.markdown("<div class='section-title'>Scatter Matrix</div>", unsafe_allow_html=True)
    sample2 = filtered.sample(min(200, len(filtered)), random_state=1)
    fig = px.scatter_matrix(sample2,
        dimensions=['Applicant_Income','Loan_Amount','EMI','Credit_History'],
        color='Loan_Status', color_discrete_map={'Y':'#6366f1','N':'#ec4899'},
        opacity=.6)
    fig.update_traces(diagonal_visible=False, marker_size=4)
    fig.update_layout(**PLOTLY_LAYOUT, height=520)
    st.plotly_chart(fig, use_container_width=True)

# ── Footer ──
st.markdown("""
<div style='text-align:center;color:#1e293b;font-size:.75rem;margin-top:40px;padding:20px 0;border-top:1px solid #1e293b'>
  LoanIQ &nbsp;·&nbsp; Built with Streamlit + Plotly &nbsp;·&nbsp; Demo purposes only
</div>
""", unsafe_allow_html=True)
