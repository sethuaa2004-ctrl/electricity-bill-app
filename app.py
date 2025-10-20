import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="SmartBill AI - Electricity Predictor",
    layout="wide"
)

st.title("⚡ SmartBill AI - Electricity Bill Predictor")
st.success("✅ App is Live!")

tab1, tab2, tab3 = st.tabs(["📊 Predictions", "🌍 Carbon Calculator", "ℹ️ About"])

with tab1:
    st.subheader("📊 Prediction Inputs")
    tariff = st.number_input("Price per kWh ($)", min_value=0.05, max_value=1.0, value=0.12, step=0.01)
    fixed = st.number_input("Fixed Monthly Charge ($)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    co2 = st.number_input("CO2 per kWh (kg)", min_value=0.1, max_value=2.0, value=0.82, step=0.01)
    
    st.write("### Sample Predictions (Next 6 Months)")
    data = {
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'kWh': [350, 360, 370, 365, 375, 380],
        'Bill ($)': [50+tariff*350, 50+tariff*360, 50+tariff*370, 50+tariff*365, 50+tariff*375, 50+tariff*380]
    }
    df = pd.DataFrame(data)
    st.dataframe(df)

with tab2:
    st.subheader("🌍 Carbon Reduction Calculator")
    led = st.slider("LED bulbs", 0, 20, 5)
    trees = st.slider("Trees planted", 0, 50, 5)
    total = led * 10 + trees * 20
    st.metric("Total CO2 Saved", f"{total} kg/year")

with tab3:
    st.markdown("""
    ### About SmartBill AI
    An AI system for electricity bill predictions and carbon tracking.
    """)
