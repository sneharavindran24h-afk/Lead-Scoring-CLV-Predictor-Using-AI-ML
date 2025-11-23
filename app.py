import streamlit as st
import pickle
import numpy as np

# Load Models
lead_model = pickle.load(open("lead_model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
clv_model = pickle.load(open("clv_model.pkl", "rb"))

st.set_page_config(page_title="AI Lead Scoring & CLV", layout="wide")

st.title("🤖 AI Lead Scoring & CLV Prediction")

st.sidebar.header("Input Features")

engagement = st.sidebar.slider("Engagement (0-100)", 0, 100, 50)
visits = st.sidebar.number_input("Website Visits", 0, 200, 10)
purchase_value = st.sidebar.number_input("Avg Purchase Value (₹)", 1000, 500000, 20000)
purchase_freq = st.sidebar.slider("Purchase Frequency", 1, 12, 3)
retention_rate = st.sidebar.slider("Retention Rate", 0.0, 1.0, 0.65)
past_purchase = st.sidebar.number_input("Past Purchase (₹)", 0, 500000, 30000)

# Prepare Input
X = np.array([[engagement, visits, purchase_value, purchase_freq, retention_rate, past_purchase]])

# Predictions
lead_pred_encoded = lead_model.predict(X)[0]
lead_pred = label_encoder.inverse_transform([lead_pred_encoded])[0]
clv_pred = clv_model.predict(X)[0]

# Output
st.subheader("📊 AI Predictions")
st.write(f"### 🔹 Lead Quality: **{lead_pred}**")
st.write(f"### 🔹 Predicted CLV: **₹ {clv_pred:,.2f}**")
