import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# --- Page Config ---
st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.title('📈 AI Stock Market Prediction Pro')

# --- Sidebar Inputs ---
st.sidebar.header("User Input")
ticker = st.sidebar.text_input('Enter Stock Ticker', 'AAPL')
start_date = "2015-01-01"
end_date = datetime.now().strftime('%Y-%m-%d')

# --- Data Loading ---
@st.cache_data
def load_data(symbol):
    data = yf.download(symbol, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(ticker)
data_load_state.text('Loading data... done!')

# --- Visualizations ---
st.subheader(f'Raw Data for {ticker}')
st.write(data.tail())

st.subheader('Closing Price vs Time Chart')
fig1 = plt.figure(figsize=(12,6))
plt.plot(data.Date, data.Close, 'b')
plt.xlabel('Date')
plt.ylabel('Price')
st.pyplot(fig1)

# Moving Averages
st.subheader('Price vs MA100 vs MA200')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig2 = plt.figure(figsize=(12,6))
plt.plot(data.Close, 'b', label='Original Price')
plt.plot(ma100, 'r', label='MA100')
plt.plot(ma200, 'g', label='MA200')
plt.legend()
st.pyplot(fig2)

# --- Model Prediction ---
st.subheader('Prediction Results')
try:
    # Load the compressed pipeline and model
    model = joblib.load('stock_model.sav')
    scaler = joblib.load('scaler.sav')
    
    st.success("Model and Scaler loaded successfully from .sav files!")
    
    # Preprocessing logic for latest 60 days to predict tomorrow
    last_60_days = data['Close'].tail(60).values.reshape(-1,1)
    last_60_days_scaled = scaler.transform(last_60_days)
    
    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    
    st.metric(label=f"Predicted Next Close Price for {ticker}", value=f"${pred_price[0][0]:.2f}")

except Exception as e:
    st.warning("Upload 'stock_model.sav' and 'scaler.sav' to the GitHub repo to see predictions.")
    st.error(f"Error details: {e}")
