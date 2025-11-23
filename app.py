import streamlit as st
import pickle
import numpy as np
import plotly.graph_objs as go
import pandas as pd

# -------------------------------
# LOAD MODELS
# -------------------------------
lead_model = pickle.load(open("lead_model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
clv_model = pickle.load(open("clv_model.pkl", "rb"))

# -------------------------------
# STREAMLIT PAGE SETTINGS
# -------------------------------
st.set_page_config(
    page_title="AI Lead Scoring & CLV Predictor",
    page_icon="📈",
    layout="wide"
)

# -------------------------------
# CUSTOM CSS FOR BEAUTY
# -------------------------------
st.markdown("""
<style>
.big-font {
    font-size:32px !important;
    font-weight:700;
}
.card {
    padding:25px;
    border-radius:15px;
    color:white;
    text-align:center;
}
.high { background-color:#2ECC71; }
.medium { background-color:#F4D03F; }
.low { background-color:#E74C3C; }
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #e6e6e6;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER SECTION
# -------------------------------
st.markdown("<h1 class='big-font' style='text-align:center;color:#2E86C1;'>AI-Driven Lead Scoring & CLV Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:18px;'>Smart ML models to estimate Lead Quality and Customer Lifetime Value</p>", unsafe_allow_html=True)
st.write("---")

# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
st.sidebar.header("🔧 Input Parameters")

engagement = st.sidebar.slider("Engagement (0–100)", 0, 100, 50)
visits = st.sidebar.number_input("Website Visits", 0, 200, 10)
purchase_value = st.sidebar.number_input("Avg Purchase Value (₹)", 1000, 500000, 20000)
purchase_freq = st.sidebar.slider("Purchase Frequency", 1, 12, 3)
retention_rate = st.sidebar.slider("Retention Rate", 0.0, 1.0, 0.65)
past_purchase = st.sidebar.number_input("Past Purchase History (₹)", 0, 500000, 30000)

X = np.array([[engagement, visits, purchase_value, purchase_freq, retention_rate, past_purchase]])
X_clv = np.array([[purchase_value, purchase_freq, retention_rate]])

# -------------------------------
# AI PREDICTIONS
# -------------------------------
lead_pred_encoded = lead_model.predict(X)[0]
lead_pred = label_encoder.inverse_transform([lead_pred_encoded])[0]
clv_pred = clv_model.predict(X_clv)[0]

# -------------------------------
# DISPLAY PREDICTION CARDS
# -------------------------------
lead_color_class = "high" if lead_pred=="High" else "medium" if lead_pred=="Medium" else "low"

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"<div class='card {lead_color_class}'><h2>Lead Score</h2><h1>{lead_pred}</h1></div>",
        unsafe_allow_html=True)

with col2:
    st.markdown(
        f"<div class='card' style='background:#3498DB;'><h2>Predicted CLV</h2><h1>₹ {clv_pred:,.2f}</h1></div>",
        unsafe_allow_html=True)

st.write("---")

# -------------------------------
# FEATURE IMPORTANCE CHART
# -------------------------------
st.subheader("📊 Feature Importance (Lead Scoring Model)")

importance = lead_model.feature_importances_
feature_names = ["Engagement", "Visits", "Purchase Value", "Purchase Freq", "Retention Rate", "Past Purchase"]

fig = go.Figure(go.Bar(
    x=importance,
    y=feature_names,
    orientation='h'
))
fig.update_layout(
    height=400,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Importance Score",
    yaxis_title="Feature"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# CLV SENSITIVITY CURVE
# -------------------------------
st.subheader("📈 CLV Sensitivity Curve")

rr_range = np.linspace(0.1, 0.95, 50)
clv_curve = [(purchase_value * purchase_freq * rr) / (1 - rr) for rr in rr_range]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=rr_range,
    y=clv_curve,
    mode="lines",
    line=dict(width=4)
))
fig2.update_layout(
    height=400,
    xaxis_title="Retention Rate",
    yaxis_title="CLV"
)

st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# RECOMMENDATION ENGINE
# -------------------------------
st.subheader("🎯 Recommended Action")

if lead_pred == "High":
    st.success("🚀 **Top Priority Lead** → Assign salesperson + offer high-value proposal.")
elif lead_pred == "Medium":
    st.info("📩 **Warm Lead** → Send email nurturing sequence + targeted product demo.")
else:
    st.warning("⏳ **Low Priority Lead** → Put into long-term drip campaign.")

st.write("---")
st.caption("Built using Streamlit • Machine Learning • Random Forest Models")
