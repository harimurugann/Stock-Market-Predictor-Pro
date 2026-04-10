import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

from streamlit_autorefresh import st_autorefresh
import datetime

# Refresh the app every 60 seconds to update live data
count = st_autorefresh(interval=60000, key="fizzbuzzcounter")

# Display Live Time in Sidebar or Header
now = datetime.datetime.now().strftime("%H:%M:%S")
st.sidebar.markdown(f"### 🕒 Server Time: {now}")

# --- 0. App Configuration ---
st.set_page_config(page_title="Live Stock Tracker & Predictor", layout="wide")

# --- 1. Live Tracking Feature (Auto-refresh every 60 seconds) ---
st.sidebar.header("Live Tracking Settings")
auto_refresh = st.sidebar.checkbox("Enable Live Tracking", value=False)

if auto_refresh:
    # Refresh every 60,000 milliseconds (1 minute)
    st_autorefresh(interval=60000, key="stock_refresh")
    st.sidebar.success("Live Tracking Active (Updating every 1 min)")

# --- 2. Sidebar Inputs ---
popular_symbols = ["NVDA", "AAPL", "TSLA", "RELIANCE.NS", "TCS.NS", "BTC-USD"]
choice = st.sidebar.selectbox("Select Asset", ["Type my own..."] + popular_symbols)
symbol = st.sidebar.text_input("Ticker", "MSFT").upper() if choice == "Type my own..." else choice

# --- 3. Main Dashboard ---
st.title("📈 Live Stock Tracker & AI Predictor")

# Fetching Data
with st.spinner('Fetching live market data...'):
    df = yf.download(symbol, period="2y", interval="1d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

if not df.empty and len(df) > 100:
    # Feature Engineering
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Target'] = df['Close'].shift(-1)
    df_clean = df.dropna()

    # Model Training Logic
    X = df_clean[['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'MA50']]
    y = df_clean['Target']
    
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    model_pipeline.fit(X, y)

    # --- Metrics Dashboard ---
    latest_data = df.tail(1)
    current_price = float(latest_data['Close'].iloc[0])
    prev_close = float(df['Close'].iloc[-2])
    price_change = current_price - prev_close
    percent_change = (price_change / prev_close) * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f} ({percent_change:.2f}%)")
    m2.metric("Market Status", "LIVE" if auto_refresh else "STATIC")
    m3.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))

    # --- Candlestick Chart ---
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                         low=df['Low'], close=df['Close'], name='Market')])
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='orange')))
    fig.update_layout(template='plotly_dark', height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- Prediction Result ---
    prediction_input = latest_data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'MA50']]
    pred_val = model_pipeline.predict(prediction_input)[0]

    st.subheader(f"🎯 AI Forecast for Tomorrow")
    p1, p2 = st.columns(2)
    p1.info(f"Predicted Closing Price: **${pred_val:.2f}**")
    
    diff = pred_val - current_price
    if diff > 0:
        p2.success(f"Signal: **BULLISH** (Expected +${diff:.2f})")
    else:
        p2.warning(f"Signal: **BEARISH** (Expected -${abs(diff):.2f})")

else:
    st.error("Data not available for the selected symbol.")

st.caption("Auto-refresh is active. The AI retrains every minute with the latest market tick.")
