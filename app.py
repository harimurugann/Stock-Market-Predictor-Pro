import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="AI Stock Predictor Pro", layout="wide")
st.title('📈 AI Stock Market Prediction Pro')

ticker = st.sidebar.text_input('Enter Stock Ticker', 'AAPL').upper()

# Data Fetching
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d')

@st.cache_data
def load_data(symbol):
    data = yf.download(symbol, start=start_date, end=end_date)
    if data.empty: return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    return data

data = load_data(ticker)

if data is not None:
    st.subheader(f'Price Chart for {ticker}')
    st.line_chart(data.set_index('Date')['Close'])

    # PREDICTION LOGIC
    try:
        # Load .h5 model using Keras
        model = tf.keras.models.load_model('stock_model.h5')
        scaler = joblib.load('scaler.sav')

        recent_data = data['Close'].tail(60).values.reshape(-1,1)
        scaled_data = scaler.transform(recent_data)
        
        X_test = np.array([scaled_data])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        pred = model.predict(X_test)
        pred_final = scaler.inverse_transform(pred)

        st.success(f"Predicted Next Close Price: ${pred_final[0][0]:.2f}")
    except Exception as e:
        st.warning(f"Model Loading Error: {e}")
