import streamlit as st
import joblib
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Load the saved pipeline
# Ensure 'full_stock_pipeline.sav' is in the same directory
try:
    model = joblib.load('full_stock_pipeline.sav')
except:
    st.error("Model file not found. Please run the training notebook first.")

st.set_page_config(page_title="Stock Predictor Pro", layout="wide")

st.title("📈 Advanced Stock Market Prediction Dashboard")

# Sidebar for user input
st.sidebar.header("User Input")
symbol = st.sidebar.text_input("Enter Stock Ticker", "NVDA").upper()
days_to_plot = st.sidebar.slider("Days of history to show", 30, 365, 90)

if st.sidebar.button("Fetch and Predict"):
    with st.spinner('Fetching live data...'):
        # Fetching data for visualization and prediction
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_to_plot + 60) # Extra buffer for MA calculation
        df = yf.download(symbol, start=start_date, end=end_date)

    if not df.empty:
        # Feature Engineering for prediction
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # Displaying Key Metrics
        latest_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        price_diff = latest_price - prev_price

        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${latest_price:.2f}", f"{price_diff:.2f}")
        
        # Visualization Section
        st.subheader(f"{symbol} Live Analysis Graph")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                     low=df['Low'], close=df['Close'], name='Market Data'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='MA 20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], line=dict(color='blue', width=1), name='MA 50'))
        
        fig.update_layout(title=f'{symbol} Price Chart', yaxis_title='Stock Price (USD)', height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Prediction Logic
        # Get the latest features for the model
        latest_features = df.tail(1)[['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'MA50']]
        
        if not latest_features.isnull().values.any():
            prediction = model.predict(latest_features)
            
            st.success(f"### 🎯 Predicted Next Day Close: **${prediction[0]:.2f}**")
            
            # Additional logic for Buy/Sell signal
            if prediction[0] > latest_price:
                st.info("💡 Advice: Predicted price is higher. Potential 'BUY' signal.")
            else:
                st.warning("💡 Advice: Predicted price is lower. Potential 'SELL/HOLD' signal.")
        else:
            st.warning("Not enough data to calculate Moving Averages for prediction. Try a longer timeframe.")
    else:
        st.error("Invalid Ticker or No Data Found.")
