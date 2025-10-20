import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#from tensorflow.keras.models import load_model
import os

#import google.generativeai as genai

st.set_page_config(
    page_title="SmartBill AI - Electricity Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ SmartBill AI - Electricity Bill Predictor")
st.markdown("Predict your electricity consumption, bills, and carbon emissions using AI")

@st.cache_resource
def load_model_and_scaler():
    try:
        #model = load_model('lstm_model.h5')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.error("❌ Model or scaler files not found!")
    st.info("Please ensure 'lstm_model.h5' and 'scaler.pkl' are in the same directory as this app.")
    st.stop()

st.success("✅ Model loaded successfully!")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Predictions", "💬 SmartBill Assistant", "🌍 Carbon Calculator", "ℹ️ About"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Prediction Inputs")
        
        if "historical_data" not in st.session_state:
            st.session_state.historical_data = []
        
        with st.expander("📤 Upload Monthly Bills (To Improve Predictions)", expanded=False):
            st.write("Upload your actual monthly consumption data to improve prediction accuracy!")
            
            upload_col1, upload_col2 = st.columns(2)
            
            with upload_col1:
                uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
                
                if uploaded_file is not None:
                    try:
                        uploaded_df = pd.read_csv(uploaded_file)
                        st.write("**Preview of uploaded data:**")
                        st.dataframe(uploaded_df.head())
                        
                        if st.button("✅ Add to Historical Data"):
                            st.session_state.historical_data = uploaded_df.to_dict('records')
                            st.success(f"✅ Added {len(uploaded_df)} months of data!")
                            
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
            
            with upload_col2:
                st.write("**CSV Format Required:**")
                st.code("""date,kWh,bill_amount
2024-01-01,350,50
2024-02-01,360,52""")
                template_df = pd.DataFrame({
                    'date': ['2024-01-01', '2024-02-01', '2024-03-01'],
                    'kWh': [350, 360, 370],
                    'bill_amount': [50, 52, 54]
                })
                csv = template_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV Template",
                    data=csv,
                    file_name="bill_template.csv",
                    mime="text/csv"
                )
            
            if st.session_state.historical_data:
                st.write("**Current Historical Data:**")
                historical_df = pd.DataFrame(st.session_state.historical_data)
                st.dataframe(historical_df)
                
                if st.button("🗑️ Clear Historical Data"):
                    st.session_state.historical_data = []
                    st.rerun()
        
        predictions_df = None
        if os.path.exists('predictions.csv'):
            predictions_df = pd.read_csv('predictions.csv')
            st.info(f"✓ Loaded {len(predictions_df)} months of future predictions")

        col_a, col_b = st.columns(2)
        
        with col_a:
            st.write("**Electricity Tariff**")
            tariff_per_kwh = st.number_input(
                "Price per kWh ($)",
                min_value=0.05,
                max_value=1.0,
                value=0.12,
                step=0.01
            )
        
        with col_b:
            st.write("**Additional Charges**")
            fixed_charge = st.number_input(
                "Fixed Monthly Charge ($)",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=1.0
            )

        st.write("**Carbon Emissions Factor**")
        co2_factor = st.number_input(
            "CO2 per kWh (kg)",
            min_value=0.1,
            max_value=2.0,
            value=0.82,
            step=0.01
        )

        st.write("**Prediction Mode**")
        mode = st.radio("Choose prediction method:", ["View All Forecasts", "Custom Input"])

        if mode == "View All Forecasts":
            if predictions_df is not None:
                st.subheader("📈 Next 6 Months Forecasts")
                
                predictions_df['predicted_bill'] = (
                    predictions_df['predicted_kWh'] * tariff_per_kwh + fixed_charge
                )
                predictions_df['CO2_kg'] = predictions_df['predicted_kWh'] * co2_factor
                
                cols = st.columns(4)
                with cols[0]:
                    avg_kwh = predictions_df['predicted_kWh'].mean()
                    st.metric("Avg kWh/month", f"{avg_kwh:.2f}")
                with cols[1]:
                    avg_bill = predictions_df['predicted_bill'].mean()
                    st.metric("Avg Bill", f"${avg_bill:.2f}")
                with cols[2]:
                    total_co2 = predictions_df['CO2_kg'].sum()
                    st.metric("Total CO2 (6mo)", f"{total_co2:.2f} kg")
                with cols[3]:
                    co2_tons = total_co2 / 1000
                    st.metric("Total CO2", f"{co2_tons:.3f} tons")
                
                st.dataframe(
                    predictions_df[[
                        'date', 'predicted_kWh', 'predicted_bill', 'CO2_kg'
                    ]].assign(
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
                ax1.fill_between(range(len(predictions_df)), predictions_df['predicted_kWh'], 
                                alpha=0.3, color='#1f77b4')
                ax1.set_xlabel('Month Ahead')
                ax1.set_ylabel('kWh')
                ax1.set_title('Predicted Electricity Consumption')
                ax1.grid(True, alpha=0.3)
                
                ax2.bar(range(len(predictions_df)), predictions_df['predicted_bill'], 
                       color='#ff7f0e', alpha=0.7)
                ax2.set_xlabel('Month Ahead')
                ax2.set_ylabel('Bill Amount ($)')
                ax2.set_title('Predicted Monthly Bills')
                ax2.grid(True, alpha=0.3, axis='y')
                
                ax3.bar(range(len(predictions_df)), predictions_df['CO2_kg'], 
                       color='#2ca02c', alpha=0.7)
                ax3.set_xlabel('Month Ahead')
                ax3.set_ylabel('CO2 (kg)')
                ax3.set_title('Carbon Emissions Forecast')
                ax3.grid(True, alpha=0.3, axis='y')
                
                labels = ['Bills', 'Fixed Charges']
                bills_var = (predictions_df['predicted_bill'] - fixed_charge).sum()
                fixed_total = fixed_charge * len(predictions_df)
                sizes = [bills_var, fixed_total]
                colors = ['#ff7f0e', '#d62728']
                ax4.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
                ax4.set_title('6-Month Cost Breakdown')
                
                st.pyplot(fig)
            else:
                st.warning("Predictions file not found. Train the model first.")

        else:
            st.subheader("🔮 Make Custom Prediction")
            
            if st.session_state.historical_data:
                historical_df = pd.DataFrame(st.session_state.historical_data)
                if 'kWh' in historical_df.columns:
                    last_12_months_data = historical_df['kWh'].tail(12).tolist()
                    
                    if len(last_12_months_data) == 12:
                        st.info("✅ Using your uploaded monthly data for prediction!")
                        st.write(f"**Last 12 months:** {', '.join([f'{x:.0f}' for x in last_12_months_data])}")
                        
                        if st.button("🔮 Predict Next Month (Using Uploaded Data)", use_container_width=True, type="primary"):
                            try:
                                scaled_input = scaler.transform(np.array(last_12_months_data).reshape(-1, 1))
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
                                
                                st.write("**Bill Breakdown:**")
                                st.write(f"- Consumption: {prediction_kwh:.2f} kWh × ${tariff_per_kwh}/kWh = ${prediction_kwh * tariff_per_kwh:.2f}")
                                st.write(f"- Fixed Charge: ${fixed_charge:.2f}")
                                st.write(f"- **Total: ${predicted_bill:.2f}**")
                                
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                    else:
                        st.warning(f"⚠️ You have {len(last_12_months_data)} months. Please upload 12 months.")
            
            st.divider()
            st.write("**Or enter manually:**")
            
            last_12_months = st.text_area(
                "Enter last 12 months of kWh (comma-separated):",
                value="",
                height=100,
                placeholder="e.g., 350, 360, 370, 380, ..."
            )
            
            if st.button("🔮 Predict Next Month (Manual Entry)", use_container_width=True, type="primary"):
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
                        
                        st.write("**Bill Breakdown:**")
                        st.write(f"- Consumption: {prediction_kwh:.2f} kWh × ${tariff_per_kwh}/kWh = ${prediction_kwh * tariff_per_kwh:.2f}")
                        st.write(f"- Fixed Charge: ${fixed_charge:.2f}")
                        st.write(f"- **Total: ${predicted_bill:.2f}**")
                        
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
        
        **Tips:**
        - Get tariff from your bill
        - Enter accurate data
        - Update monthly
        """)

with tab2:
    st.subheader("💬 SmartBill AI Assistant")
    st.write("Ask me anything about electricity bills, consumption, or carbon emissions!")
    
    with st.sidebar:
        st.write("---")
        st.subheader("🔑 API Configuration")
        api_choice = st.radio("Choose AI Provider:", ["Google Gemini", "OpenAI"])
        
        if api_choice == "Google Gemini":
            gemini_key = st.text_input("Enter your Google Gemini API Key:", type="password")
            st.caption("📚 [Get Gemini API Key](https://aistudio.google.com/app/apikey)")
        else:
            openai_key = st.text_input("Enter your OpenAI API Key:", type="password")
            st.caption("💳 [Get OpenAI API Key](https://platform.openai.com/api-keys)")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div style="text-align: right; margin: 10px 0;">
                    <div style="background-color: #0066cc; color: white; padding: 12px 16px; border-radius: 12px; display: inline-block; max-width: 70%;">
                        {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="text-align: left; margin: 10px 0;">
                    <div style="background-color: #e8e8e8; color: black; padding: 12px 16px; border-radius: 12px; display: inline-block; max-width: 70%;">
                        {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.divider()
    user_input = st.text_input("Type your question here...", placeholder="e.g., How can I reduce my electricity bill?")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        try:
            if api_choice == "Google Gemini" and gemini_key:
                #genai.configure(api_key=gemini_key)
                model_gem = genai.GenerativeModel('gemini-pro')
                
                system_prompt = "You are SmartBill AI Assistant, an expert in electricity bills and energy efficiency. Help users understand consumption, reduce bills, and track carbon emissions."
                full_prompt = f"{system_prompt}\n\nUser: {user_input}"
                response = model_gem.generate_content(full_prompt)
                bot_reply = response.text
            else:
                bot_reply = "Please enter a valid API key in the sidebar!"
            
            st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

with tab3:
    st.subheader("🌍 Carbon Reduction Calculator")
    st.write("See how much CO2 you can save!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💡 Lighting & Appliances")
        led_bulbs = st.slider("LED bulbs installed", 0, 20, 5)
        led_savings = led_bulbs * 10
        st.metric("CO2 Saved (LED)", f"{led_savings} kg/year")
        
        smart_thermostat = st.checkbox("Smart Thermostat?")
        thermostat_savings = 1000 if smart_thermostat else 0
        st.metric("CO2 Saved (Thermostat)", f"{thermostat_savings} kg/year")
    
    with col2:
        st.markdown("### 🌳 Habits & Transport")
        trees = st.slider("Trees planted", 0, 50, 5)
        trees_savings = trees * 20
        st.metric("CO2 Absorbed (Trees)", f"{trees_savings} kg/year")
        
        transport_days = st.slider("Public transport days/week", 0, 7, 3)
        transport_savings = transport_days * 5 * 52
        st.metric("CO2 Saved (Transport)", f"{transport_savings} kg/year")
    
    total = led_savings + thermostat_savings + trees_savings + transport_savings
    st.metric("Total CO2 Saved", f"{total:.0f} kg/year")

with tab4:
    st.subheader("ℹ️ About SmartBill AI")
    st.markdown("""
    ### What is SmartBill AI?
    An AI system that predicts electricity consumption using LSTM neural networks.
    
    ### Features
    - AI-powered predictions
    - Bill estimation
    - Carbon tracking
    - Smart chatbot
    - Monthly updates
    
    ### Technology
    - LSTM Neural Network
    - Streamlit frontend
    - TensorFlow backend
    - Google Gemini / OpenAI chatbot
    """)
