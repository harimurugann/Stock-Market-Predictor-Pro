import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(page_title="AI Stock Predictor Pro", layout="wide")
st.title('📈 AI Stock Market Prediction Pro')
st.markdown("Developed by **Harimurugan**")

# Sidebar
ticker = st.sidebar.text_input('Enter Stock Ticker (e.g., AAPL, TSLA)', 'AAPL').upper()

# Data Fetching logic
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d')

@st.cache_data
def load_data(symbol):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty: return None
        # Fix for new yfinance version columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.reset_index(inplace=True)
        return data
    except: return None

data = load_data(ticker)

if data is not None:
    # Display Chart
    st.subheader(f'Stock Price Analysis: {ticker}')
    st.line_chart(data.set_index('Date')['Close'])

   # --- AI PREDICTION SECTION ---
    st.divider()
    try:
        import os
        model_path = 'stock_model.keras'
        
        if os.path.exists(model_path):
            # Load model with Keras 3 compatibility
            model = tf.keras.models.load_model(model_path, compile=False)
            scaler = joblib.load('scaler.sav')

            # Prepare Data (Last 60 days)
            last_60_days = data['Close'].tail(60).values.reshape(-1, 1)
            scaled_data = scaler.transform(last_60_days)
            
            X_test = np.array([scaled_data])
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            # AI Prediction
            pred = model.predict(X_test)
            final_pred = scaler.inverse_transform(pred)

            # Success Display
            st.balloons()
            st.success(f"### 🤖 AI Prediction for Next Day Close: **${final_pred[0][0]:.2f}**")
        else:
            st.error("Model file 'stock_model.keras' not found in GitHub!")
            
    except Exception as e:
        st.error(f"Technical Error: {e}")
