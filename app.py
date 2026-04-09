import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf

# --- Page Config ---
st.set_page_config(page_title="AI Stock Predictor Pro", layout="wide")
st.title('📈 AI Stock Market Prediction Pro')
st.markdown("Developed by **Harimurugan** | Real-time Stock Analysis")

# --- Sidebar ---
ticker = st.sidebar.text_input('Enter Stock Ticker', 'AAPL').upper()

# --- Data Fetching (Fixed Version) ---
@st.cache_data
def load_data(symbol):
    try:
        # Fetching data
        df = yf.download(symbol, start='2015-01-01', end=datetime.now().strftime('%Y-%m-%d'))
        
        if df.empty:
            return None
        
        # MUKKIYAMAANA STEP: Fix for Multi-index columns in new yfinance version
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        return None

data = load_data(ticker)

if data is None:
    st.error(f"Could not find data for ticker: {ticker}. Please check the symbol.")
else:
    # --- Visualizations ---
    st.subheader(f'Price History of {ticker}')
    fig1 = plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'], 'b')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(fig1)

    # --- Prediction ---
    try:
        # Load model and scaler safely
        # Keras 3 might need native loading
        model = tf.keras.models.load_model('stock_model.sav')
        scaler = joblib.load('scaler.sav')
        
        last_60_days = data['Close'].tail(60).values.reshape(-1, 1)
        scaled_data = scaler.transform(last_60_days)
        
        X_test = []
        X_test.append(scaled_data)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        prediction = model.predict(X_test)
        final_price = scaler.inverse_transform(prediction)
        
        st.divider()
        st.subheader("🚀 AI Prediction")
        st.metric(label=f"Next Predicted Close for {ticker}", value=f"${final_price[0][0]:.2f}")
        
    except Exception as e:
        st.warning("Prediction is currently unavailable due to model loading. Checking charts...")
