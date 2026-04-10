import streamlit as st
import joblib
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 0. App Configuration ---
st.set_page_config(page_title="Stock Predictor Pro", layout="wide")

# --- 1. Load the Saved Pipeline ---
@st.cache_resource
def load_model():
    # Loading the compressed pipeline saved as .sav
    return joblib.load('full_stock_pipeline.sav')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- 2. Sidebar Configuration ---
st.sidebar.header("Stock Selection")
symbol = st.sidebar.text_input("Enter Ticker Symbol", "NVDA").upper()
time_period = st.sidebar.selectbox("Select History Period", ["3mo", "6mo", "1y", "2y"])

# --- 3. Main Dashboard ---
st.title("📈 Live Stock Market Prediction Dashboard")
st.markdown(f"Currently analyzing: **{symbol}**")

if st.sidebar.button("Get Live Analysis"):
    with st.spinner('Fetching live data from Yahoo Finance...'):
        # Fetch data
        data = yf.download(symbol, period=time_period, interval="1d")
    
    if not data.empty:
        # --- 4. Feature Engineering (Live) ---
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        
        # Fixing potential Multi-Index or Series issues for Metrics
        latest_price = float(data['Close'].iloc[-1])
        prev_price = float(data['Close'].iloc[-2])
        price_diff = latest_price - prev_price

        # --- 5. Metrics Display ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${latest_price:.2f}", f"{price_diff:.2f}")
        col2.metric("20-Day MA", f"${float(data['MA20'].iloc[-1]):.2f}")
        col3.metric("50-Day MA", f"${float(data['MA50'].iloc[-1]):.2f}")

        # --- 6. Interactive Visualization (Plotly) ---
        st.subheader("Interactive Market Chart")
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ))
        
        # Adding Moving Averages
        fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], line=dict(color='orange', width=1.5), name='MA 20'))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], line=dict(color='cyan', width=1.5), name='MA 50'))

        fig.update_layout(
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            height=500,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- 7. Prediction Logic ---
        # Preparing the input for the model
        latest_row = data.tail(1)[['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'MA50']]
        
        if not latest_row.isnull().values.any():
            prediction = model.predict(latest_row)
            pred_val = float(prediction[0])
            
            st.divider()
            st.subheader("🎯 Model Prediction")
            
            p_col1, p_col2 = st.columns(2)
            p_col1.write(f"### Next Trading Day's Predicted Close: **${pred_val:.2f}**")
            
            # Simple Buy/Sell logic
            if pred_val > latest_price:
                p_col2.success("💡 Signal: BULLISH (Predicted price is higher)")
            else:
                p_col2.warning("💡 Signal: BEARISH (Predicted price is lower)")
        else:
            st.warning("Not enough data to calculate all features for prediction.")
            
    else:
        st.error("Invalid ticker or data unavailable. Please check the symbol.")

# --- 8. Footer ---
st.caption("Data source: Yahoo Finance | Built with Streamlit & Scikit-Learn")
