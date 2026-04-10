import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from datetime import datetime, timedelta

# --- 0. App Configuration ---
st.set_page_config(page_title="Dynamic Stock Predictor AI", layout="wide")

# --- 1. Sidebar for Live Inputs ---
st.sidebar.header("AI Model Settings")
popular_symbols = ["NVDA", "AAPL", "TSLA", "RELIANCE.NS", "TCS.NS", "BTC-USD", "ETH-USD"]
choice = st.sidebar.selectbox("Choose a stock or type below", ["Type my own..."] + popular_symbols)

if choice == "Type my own...":
    symbol = st.sidebar.text_input("Enter Ticker Symbol", "MSFT").upper()
else:
    symbol = choice

# --- 2. Main Dashboard ---
st.title("📈 Real-time Trained Stock Predictor")
st.markdown(f"Currently Analyzing and Training for: **{symbol}**")

if st.sidebar.button("Train AI & Predict"):
    with st.spinner(f'Fetching data and training AI model for {symbol}...'):
        # Step 1: Fetch 2 years of data for training
        df = yf.download(symbol, period="2y", interval="1d")

        # Fix for Multi-index columns in newer yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    if not df.empty and len(df) > 100:
        # --- 3. Feature Engineering ---
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['Target'] = df['Close'].shift(-1) # Predicting next day's price
        df.dropna(inplace=True)

        # Splitting Features and Target
        X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'MA50']]
        y = df['Target']

        # --- 4. Live Training Logic ---
        model_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        # Training the model on the spot
        model_pipeline.fit(X, y)

        # Saving the live-trained model
        joblib.dump(model_pipeline, 'live_stock_model.sav', compress=3)

        # --- 5. Metrics & Visualization ---
        latest_data = df.tail(1)
        # Getting the last close price safely
        current_price = float(latest_data['Close'].iloc[0])

        col1, col2 = st.columns(2)
        col1.metric("Live Market Price", f"${current_price:.2f}")

        # Plotly Candlestick Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ))
        
        # Adding MA lines to graph
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='orange', width=1.5)))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(color='blue', width=1.5)))

        fig.update_layout(
            template='plotly_dark', 
            height=500, 
            title=f"{symbol} Price Trend & Moving Averages",
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- 6. Final Prediction ---
        # Using the last available row to predict tomorrow
        prediction_input = latest_data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'MA50']]
        prediction = model_pipeline.predict(prediction_input)
        pred_val = float(prediction[0])

        st.divider()
        st.subheader(f"🎯 AI Prediction for {symbol}")
        
        p_col1, p_col2 = st.columns(2)
        p_col1.write(f"### Next Trading Day's Predicted Close: **${pred_val:.2f}**")

        diff = pred_val - current_price
        if diff > 0:
            p_col2.success(f"📈 Bullish Trend: Expected increase of ${diff:.2f}")
        else:
            p_col2.warning(f"📉 Bearish Trend: Expected decrease of ${abs(diff):.2f}")

    else:
        st.error("Error: Not enough historical data found for this symbol. Please try a different one.")

st.caption("Note: This model is trained live on historical data. Market investments carry risks.")
