import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from datetime import datetime
import time

# --- 0. App Configuration ---
st.set_page_config(page_title="AI Stock Pro Dashboard", layout="wide")

# --- 1. Sidebar - Live Settings ---
st.sidebar.header("📡 Live Control Panel")

# Live Tracking Toggle
live_switch = st.sidebar.toggle("Enable Live Tracking", value=True)

# Live Clock in Sidebar
st.sidebar.divider()
clock_placeholder = st.sidebar.empty()

# More Famous Stocks
famous_stocks = {
    "NVIDIA": "NVDA", "Apple": "AAPL", "Tesla": "TSLA", "Microsoft": "MSFT", 
    "Google": "GOOGL", "Amazon": "AMZN", "Meta": "META", "Netflix": "NFLX",
    "Reliance": "RELIANCE.NS", "TCS": "TCS.NS", "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS", "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"
}
selected_stock_name = st.sidebar.selectbox("Select Stock", list(famous_stocks.keys()))
symbol = famous_stocks[selected_stock_name]

# Custom Ticker Option
custom_ticker = st.sidebar.text_input("Or Type Custom Ticker (e.g., TSLA)", "")
if custom_ticker:
    symbol = custom_ticker.upper()

# --- 2. Live Clock Logic ---
def update_clock():
    current_time = datetime.now().strftime("%H:%M:%S")
    clock_placeholder.markdown(f"### 🕒 Live Time: {current_time}")

update_clock()

# --- 3. Main Dashboard ---
st.title(f"📈 {selected_stock_name} AI Analytics")
m1, m2, m3 = st.columns(3)

# Function to fetch and train
def run_analysis():
    with st.spinner('Updating market data...'):
        df = yf.download(symbol, period="2y", interval="1d")
        
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Feature Engineering
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df['Target'] = df['Close'].shift(-1)
            df.dropna(inplace=True)

            X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'MA50']]
            y = df['Target']

            # Model Training
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            model.fit(X, y)

            # Latest Metrics
            current_price = float(df['Close'].iloc[-1])
            prev_price = float(df['Close'].iloc[-2])
            
            m1.metric("Current Price", f"${current_price:.2f}", f"{current_price-prev_price:.2f}")
            m2.metric("Market Status", "🟢 OPEN" if live_switch else "🔴 PAUSED")
            m3.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))

            # Interactive Graph
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='orange')))
            fig.update_layout(template='plotly_dark', height=500, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            # Prediction
            latest_input = df.tail(1)[['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'MA50']]
            prediction = model.predict(latest_input)[0]
            
            st.subheader(f"🎯 AI Tomorrow's Prediction: ${prediction:.2f}")
            if prediction > current_price:
                st.success(f"🔼 Bullish: Potential growth of ${prediction - current_price:.2f}")
            else:
                st.warning(f"🔽 Bearish: Potential drop of ${current_price - prediction:.2f}")
        else:
            st.error("Ticker not found!")

# Run logic based on Toggle
if live_switch:
    run_analysis()
    time.sleep(1) # Small delay
    st.rerun() # This will keep the clock and data moving
else:
    if st.sidebar.button("Manual Update"):
        run_analysis()
    st.info("Live Tracking is Paused. Use the toggle to resume.")
