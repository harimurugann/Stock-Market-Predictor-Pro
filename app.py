import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Page Setup
st.set_page_config(page_title="AI Stock Predictor Pro", layout="wide")
st.title('📈 AI Stock Market Prediction Pro')

# Sidebar
ticker = st.sidebar.text_input('Enter Stock Ticker', 'AAPL').upper()

# Fetch Data
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d')

@st.cache_data
def load_data(symbol):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.reset_index(inplace=True)
        return data
    except: return None

data = load_data(ticker)

if data is not None:
    # Plotting
    st.subheader(f'Stock Trend for {ticker}')
    st.line_chart(data.set_index('Date')['Close'])

    # Prediction
    try:
        # Loading .h5 model
        model = tf.keras.models.load_model('stock_model.h5')
        scaler = joblib.load('scaler.sav')

        # Prepare last 60 days
        last_60_days = data['Close'].tail(60).values.reshape(-1, 1)
        scaled_data = scaler.transform(last_60_days)
        
        X_input = np.array([scaled_data])
        X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))

        # Predict
        prediction = model.predict(X_input)
        final_price = scaler.inverse_transform(prediction)

        st.success(f"🤖 AI Prediction for Next Close: **${final_price[0][0]:.2f}**")
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Ensure stock_model.h5 and scaler.sav are in GitHub main folder.")
