# ==============================================================
# app.py — StockSight AI (Final Professional Edition)
# Features: Dynamic Tickers, Live Metrics, & Auto-Refresh
# ==============================================================

import streamlit as st  # MUST be first

# 1. PAGE CONFIG — First functional call
st.set_page_config(
    page_title="StockSight AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

import warnings
warnings.filterwarnings("ignore")

import os, time, traceback
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 2. PATH RESOLUTION & FEATURE CONFIG
# ──────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

def mpath(name: str) -> Path:
    return MODEL_DIR / name

FEATURE_COLS = [
    "Open", "High", "Low", "Volume", "SMA_5", "SMA_10", "SMA_20", "SMA_50",
    "EMA_5", "EMA_10", "EMA_20", "BB_width", "BB_upper", "BB_lower",
    "RSI_14", "MACD", "MACD_signal", "MACD_hist", "Volatility_5", "Volatility_20",
    "Momentum_5", "Momentum_10", "Close_lag_1", "Close_lag_2", "Close_lag_3", "Close_lag_5",
    "Return_lag_1", "Return_lag_2", "Return_lag_3", "Volume_ratio", "DayOfWeek", "Month", "Quarter",
]

# ──────────────────────────────────────────────────────────────
# 3. SIDEBAR CONFIGURATION (Added as per your request)
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ STOCKSIGHT AI")
    st.divider()

    st.markdown("### 🔧 Configuration")
    # Dropdown for tickers
    ticker_list = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "RELIANCE.NS", "TCS.NS", "BTC-USD"]
    ticker = st.selectbox("Select Asset Symbol", options=ticker_list, index=0)

    # Slider for History Data
    history_days = st.slider("Data History (Days)", min_value=250, max_value=1000, value=365)
    
    st.divider()
    
    # Live Tracking Toggle
    live_track = st.checkbox("Enable Live Tracking", value=True)
    if live_track:
        st.caption("🔄 Auto-refreshing every 60 seconds")

# ──────────────────────────────────────────────────────────────
# 4. DATA FETCHING (Updated for Dynamic Ticker)
# ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def get_stock_data(symbol: str, days: int) -> pd.DataFrame:
    # Converting days to yfinance period (e.g., 365d)
    period_str = f"{days}d"
    raw = yf.download(symbol, period=period_str, interval="1d", progress=False, auto_adjust=True)
    
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    
    raw.index = pd.to_datetime(raw.index)
    raw.sort_index(inplace=True)
    raw.ffill(inplace=True); raw.bfill(inplace=True)
    return raw

# ──────────────────────────────────────────────────────────────
# 5. CORE LOGIC FUNCTIONS (33-Features Preserved)
# ──────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    close, volume = d["Close"], d["Volume"]
    for w in [5, 10, 20, 50, 200]:
        d[f"SMA_{w}"] = close.rolling(w).mean()
    for w in [5, 10, 20]:
        d[f"EMA_{w}"] = close.ewm(span=w, adjust=False).mean()
    _bb_mid, _bb_std = close.rolling(20).mean(), close.rolling(20).std()
    d["BB_upper"], d["BB_lower"] = _bb_mid + 2*_bb_std, _bb_mid - 2*_bb_std
    d["BB_width"] = (d["BB_upper"] - d["BB_lower"]) / (_bb_mid + 1e-9)
    _delta = close.diff()
    _gain = _delta.clip(lower=0).rolling(14).mean()
    _loss = (-_delta.clip(upper=0)).rolling(14).mean()
    d["RSI_14"] = 100 - (100 / (1 + _gain/(_loss + 1e-9)))
    _ema12, _ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    d["MACD"] = _ema12 - _ema26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACD_signal"]
    _ret = close.pct_change()
    d["Volatility_5"], d["Volatility_20"] = _ret.rolling(5).std(), _ret.rolling(20).std()
    d["Momentum_5"], d["Momentum_10"] = close - close.shift(5), close - close.shift(10)
    for l in [1, 2, 3, 5]: d[f"Close_lag_{l}"] = close.shift(l)
    for l in [1, 2, 3]: d[f"Return_lag_{l}"] = _ret.shift(l)
    d["Volume_ratio"] = volume / (volume.rolling(10).mean() + 1e-9)
    d["DayOfWeek"], d["Month"], d["Quarter"] = d.index.dayofweek, d.index.month, d.index.quarter
    return d

def verify_feature_shape(df_eng: pd.DataFrame):
    clean_df = df_eng[FEATURE_COLS + ["Close"]].dropna()
    if clean_df.empty: return None, None
    X_latest = clean_df[FEATURE_COLS].iloc[[-1]].values
    return X_latest, clean_df

@st.cache_resource
def load_artifacts():
    try:
        p = joblib.load(mpath("stock_pipeline_compressed.sav"))
        s = joblib.load(mpath("scaler.sav"))
        m = joblib.load(mpath("pipeline_metadata.sav")) if mpath("pipeline_metadata.sav").exists() else {}
        return {"pipeline": p, "scaler": s, "metadata": m, "loaded": True}
    except: return {"loaded": False}

# ──────────────────────────────────────────────────────────────
# 6. MAIN EXECUTION
# ──────────────────────────────────────────────────────────────
artifacts = load_artifacts()

# Fetch and Process
with st.spinner(f"Updating data for {ticker}..."):
    try:
        df_raw = get_stock_data(ticker, history_days)
        df_eng = engineer_features(df_raw)
        X_live, df_clean = verify_feature_shape(df_eng)
    except Exception as e:
        st.error(f"Execution Error: {e}"); st.stop()

# Header Display
st.markdown(f"<h1 style='text-align:center;'>STOCKSIGHT AI: {ticker}</h1>", unsafe_allow_html=True)

# LIVE METRICS DISPLAY (Requested before the chart)
latest = df_raw.iloc[-1]
prev = df_raw.iloc[-2]
chg = float(latest["Close"]) - float(prev["Close"])
pct = (chg / float(prev["Close"])) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Price", f"${float(latest['Close']):.2f}", f"{chg:+.2f} ({pct:+.2f}%)")
c2.metric("Day High", f"${float(latest['High']):.2f}")
c3.metric("Day Low", f"${float(latest['Low']):.2f}")
c4.metric("Volume", f"{float(latest['Volume'])/1e6:.2f}M")

st.divider()

# CHART SECTION
st.markdown("### 📉 PRICE CHART")
fig = go.Figure(data=[go.Candlestick(x=df_raw.index, open=df_raw['Open'], high=df_raw['High'], low=df_raw['Low'], close=df_raw['Close'], name='OHLC')])
fig.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

# AI PREDICTION SECTION
st.markdown("### 🤖 AI PREDICTION")
if artifacts["loaded"] and X_live is not None:
    X_scaled = artifacts["scaler"].transform(X_live)
    pred = float(artifacts["pipeline"].predict(X_scaled)[0])
    
    current_val = float(df_raw["Close"].iloc[-1])
    diff = pred - current_val
    color = "#00ff88" if diff > 0 else "#ef4444"
    
    st.markdown(f"""
    <div style='border:1px solid {color}; border-radius:10px; padding:20px; text-align:center; background:rgba(0,0,0,0.3);'>
        <h4 style='color:{color}; margin:0;'>PREDICTED NEXT CLOSE</h4>
        <h1 style='color:{color}; font-size:3rem;'>${pred:.2f}</h1>
        <p style='color:#888;'>Direction: {"BULLISH" if diff > 0 else "BEARISH"} | Change: {diff:+.2f} ({(diff/current_val)*100:+.2f}%)</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.error("Model artifacts could not be loaded. Please check /model folder.")

# ──────────────────────────────────────────────────────────────
# 7. AUTO-REFRESH LOGIC (At the very end)
# ──────────────────────────────────────────────────────────────
if live_track:
    time.sleep(60)
    st.rerun()
