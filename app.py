import streamlit as st
import joblib
import yfinance as yf
import pandas as pd

# Load the saved pipeline
model = joblib.load('full_stock_pipeline.sav')

st.title("Stock Market Prediction App")

symbol = st.text_input("Enter Stock Ticker (e.g., TSLA, AAPL)", "NVDA")

if st.button("Predict Tomorrow's Price"):
    df = yf.download(symbol, period="60d", interval="1d")
    
    # Feature Engineering
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # Get the latest row
    latest_data = df.tail(1)[['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'MA50']]
    
    prediction = model.predict(latest_data)
    st.write(f"The predicted closing price for the next trading day is: ${prediction[0]:.2f}")
