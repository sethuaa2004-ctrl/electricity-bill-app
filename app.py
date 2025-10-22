import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os

st.set_page_config(
    page_title="SmartBill AI - Electricity Predictor",
    layout="wide"
)

st.title("⚡ SmartBill AI - Electricity Bill Predictor")

@st.cache_resource
def load_model_and_scaler():
    try:
        model = load_model('lstm_model.h5')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.error("❌ Model or scaler files not found!")
    st.stop()

st.success("✅ Model loaded successfully!")

tab1, tab2, tab3 = st.tabs(["📊 Predictions", "🌍 Carbon Calculator", "ℹ️ About"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Prediction Inputs")
        
        predictions_df = None
        if os.path.exists('predictions.csv'):
            predictions_df = pd.read_csv('predictions.csv')
            st.info(f"✓ Loaded {len(predictions_df)} months of future predictions")

        col_a, col_b = st.columns(2)
        
        with col_a:
            tariff_per_kwh = st.number_input("Price per kWh ($)", min_value=0.05, max_value=1.0, value=0.12, step=0.01)
        
        with col_b:
            fixed_charge = st.number_input("Fixed Monthly Charge ($)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)

        co2_factor = st.number_input("CO2 per kWh (kg)", min_value=0.1, max_value=2.0, value=0.82, step=0.01)

        mode = st.radio("Choose prediction method:", ["View All Forecasts", "Custom Input"])

        if mode == "View All Forecasts":
            if predictions_df is not None:
                st.subheader("📈 Next 6 Months Forecasts")
                
                predictions_df['predicted_bill'] = predictions_df['predicted_kWh'] * tariff_per_kwh + fixed_charge
                predictions_df['CO2_kg'] = predictions_df['predicted_kWh'] * co2_factor
                
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Avg kWh/month", f"{predictions_df['predicted_kWh'].mean():.2f}")
                with cols[1]:
                    st.metric("Avg Bill", f"${predictions_df['predicted_bill'].mean():.2f}")
                with cols[2]:
                    st.metric("Total CO2 (6mo)", f"{predictions_df['CO2_kg'].sum():.2f} kg")
                with cols[3]:
                    st.metric("Total CO2", f"{predictions_df['CO2_kg'].sum()/1000:.3f} tons")
                
                st.dataframe(
                    predictions_df[['date', 'predicted_kWh', 'predicted_bill', 'CO2_kg']].assign(
                        date=pd.to_datetime(predictions_df['date']).dt.strftime('%B %Y'),
                        predicted_kWh=predictions_df['predicted_kWh'].round(2),
                        predicted_bill=predictions_df['predicted_bill'].round(2),
                        CO2_kg=predictions_df['CO2_kg'].round(2)
                    ).rename(columns={
                        'date': 'Month',
                        'predicted_kWh': 'kWh',
                        'predicted_bill': 'Bill ($)',
                        'CO2_kg': 'CO2 (kg)'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8))
                
                ax1.plot(range(len(predictions_df)), predictions_df['predicted_kWh'], 
                        marker='o', linewidth=2, markersize=8, color='#1f77b4')
                ax1.fill_between(range(len(predictions_df)), predictions_df['predicted_kWh'], alpha=0.3, color='#1f77b4')
                ax1.set_xlabel('Month Ahead')
                ax1.set_ylabel('kWh')
                ax1.set_title('Predicted Electricity Consumption')
                ax1.grid(True, alpha=0.3)
                
                ax2.bar(range(len(predictions_df)), predictions_df['predicted_bill'], color='#ff7f0e', alpha=0.7)
                ax2.set_xlabel('Month Ahead')
                ax2.set_ylabel('Bill Amount ($)')
                ax2.set_title('Predicted Monthly Bills')
                ax2.grid(True, alpha=0.3, axis='y')
                
                ax3.bar(range(len(predictions_df)), predictions_df['CO2_kg'], color='#2ca02c', alpha=0.7)
                ax3.set_xlabel('Month Ahead')
                ax3.set_ylabel('CO2 (kg)')
                ax3.set_title('Carbon Emissions Forecast')
                ax3.grid(True, alpha=0.3, axis='y')
                
                labels = ['Variable Bills', 'Fixed Charges']
                bills_var = (predictions_df['predicted_bill'] - fixed_charge).sum()
                fixed_total = fixed_charge * len(predictions_df)
                sizes = [bills_var, fixed_total]
                colors = ['#ff7f0e', '#d62728']
                ax4.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
                ax4.set_title('6-Month Cost Breakdown')
                
                st.pyplot(fig)
            else:
                st.warning("Predictions file not found.")

        else:
            st.subheader("🔮 Make Custom Prediction")
            
            last_12_months = st.text_area("Enter last 12 months of kWh (comma-separated):", value="", height=100, placeholder="e.g., 350, 360, 370, 380, ...")
            
            if st.button("🔮 Predict Next Month", use_container_width=True, type="primary"):
                try:
                    values = [float(x.strip()) for x in last_12_months.split(',')]
                    if len(values) != 12:
                        st.error(f"Please enter exactly 12 values. You entered {len(values)}.")
                    else:
                        scaled_input = scaler.transform(np.array(values).reshape(-1, 1))
                        X_input = scaled_input.reshape(1, 12, 1)
                        prediction_scaled = model.predict(X_input, verbose=0)
                        prediction_kwh = scaler.inverse_transform(prediction_scaled)[0, 0]
                        
                        predicted_bill = prediction_kwh * tariff_per_kwh + fixed_charge
                        predicted_co2 = prediction_kwh * co2_factor
                        
                        st.success("✅ Prediction Complete!")
                        
                        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                        with col_p1:
                            st.metric("Next Month kWh", f"{prediction_kwh:.2f}")
                        with col_p2:
                            st.metric("Predicted Bill", f"${predicted_bill:.2f}")
                        with col_p3:
                            st.metric("CO2 Emissions", f"{predicted_co2:.2f} kg")
                        with col_p4:
                            st.metric("CO2 in Tons", f"{predicted_co2/1000:.4f} T")
                        
                        st.write(f"**Bill Breakdown:** {prediction_kwh:.2f} kWh × ${tariff_per_kwh}/kWh + ${fixed_charge:.2f} = **${predicted_bill:.2f}**")
                        
                except ValueError:
                    st.error("Invalid input! Please enter numbers separated by commas.")
    
    with col2:
        st.subheader("ℹ️ Quick Info")
        st.markdown("""
        **Features:**
        - AI forecasting
        - 6-month predictions
        - Bill estimation
        - CO2 tracking
        """)

with tab2:
    st.subheader("🌍 Carbon Reduction Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        led_bulbs = st.slider("LED bulbs installed", 0, 20, 5)
        led_savings = led_bulbs * 10
        st.metric("CO2 Saved (LED)", f"{led_savings} kg/year")
        
        smart_thermostat = st.checkbox("Smart Thermostat?")
        thermostat_savings = 1000 if smart_thermostat else 0
        st.metric("CO2 Saved (Thermostat)", f"{thermostat_savings} kg/year")
    
    with col2:
        trees = st.slider("Trees planted", 0, 50, 5)
        trees_savings = trees * 20
        st.metric("CO2 Absorbed (Trees)", f"{trees_savings} kg/year")
        
        transport_days = st.slider("Public transport days/week", 0, 7, 3)
        transport_savings = transport_days * 5 * 52
        st.metric("CO2 Saved (Transport)", f"{transport_savings} kg/year")
    
    total = led_savings + thermostat_savings + trees_savings + transport_savings
    st.metric("🎯 Total CO2 Saved", f"{total} kg/year")

with tab3:
    st.markdown("""
    ### ⚡ SmartBill AI
    An AI-powered electricity consumption prediction system using LSTM neural networks.
    
    ### 🎯 Features
    - LSTM model for time-series predictions
    - 6-month electricity forecasts
    - Bill estimation with tariff calculation
    - Carbon emission tracking
    - Interactive visualizations
    
    ### 🔧 Technology
    - Deep Learning (LSTM)
    - TensorFlow/Keras
    - Streamlit
    - Python
    """)
