import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(page_title="AI Stock Predictor Pro", layout="wide")

# Custom CSS for better look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

st.title('📈 AI Stock Market Prediction Pro')
st.markdown("Developed by **Harimurugan** | Real-time Stock Analysis")

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input('Enter Stock Ticker (e.g., AAPL, GOOGL, TSLA)', 'AAPL').upper()

# Dates for fetching data
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d') # 10 years data

# --- Data Fetching ---
@st.cache_data
def load_data(symbol):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            return None
        data.reset_index(inplace=True)
        return data
    except Exception:
        return None

data = load_data(ticker)

if data is None:
    st.error(f"Could not find data for ticker: {ticker}. Please check the symbol.")
else:
    # --- UI Layout ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f'Stock Price Trend: {ticker}')
        fig1 = plt.figure(figsize=(12, 6))
        plt.plot(data.Date, data.Close, color='#1f77b4', linewidth=2)
        plt.xlabel('Year')
        plt.ylabel('Price (USD)')
        plt.grid(True, alpha=0.3)
        st.pyplot(fig1)

    with col2:
        st.subheader("Recent Market Data")
        st.write(data[['Date', 'Close', 'Volume']].tail(10))

    # Moving Averages Analysis
    st.divider()
    st.subheader('Moving Averages (Technical Analysis)')
    ma100 = data.Close.rolling(100).mean()
    ma200 = data.Close.rolling(200).mean()
    
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(data.Close, label='Original Price', alpha=0.7)
    plt.plot(ma100, 'r', label='MA100', linewidth=1.5)
    plt.plot(ma200, 'g', label='MA200', linewidth=1.5)
    plt.legend()
    st.pyplot(fig2)

    # --- Prediction Logic ---
    st.divider()
    st.subheader('🚀 AI Prediction Results')

    try:
        # Load Model and Scaler
        model = joblib.load('stock_model.sav')
        scaler = joblib.load('scaler.sav')
        
        # Prepare Data for Prediction (Latest 60 days)
        recent_data = data['Close'].tail(60).values.reshape(-1, 1)
        scaled_recent_data = scaler.transform(recent_data)
        
        X_input = []
        X_input.append(scaled_recent_data)
        X_input = np.array(X_input)
        X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))
        
        # Predict
        predicted_scaled = model.predict(X_input)
        predicted_price = scaler.inverse_transform(predicted_scaled)
        
        current_price = data['Close'].iloc[-1]
        change = predicted_price[0][0] - current_price
        
        # Display Result in Metric
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Price", f"${current_price:.2f}")
        m2.metric("Predicted Next Close", f"${predicted_price[0][0]:.2f}", f"{change:.2f}")
        m3.metric("Stock Ticker", ticker)
        
        st.success("Prediction generated based on LSTM model patterns.")

    except FileNotFoundError:
        st.warning("⚠️ **Model Files Missing:** Please ensure 'stock_model.sav' and 'scaler.sav' are uploaded to your GitHub repository.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.sidebar.markdown("---")
st.sidebar.info("This app uses a Deep Learning LSTM model to predict future stock prices based on historical data.")
