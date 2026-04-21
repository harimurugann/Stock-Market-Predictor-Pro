# ============================================================
# app.py — StockSight AI · CLEAN & FIXED VERSION
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os, time
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────
# 1. PAGE CONFIG (Must be the first Streamlit command)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="StockSight AI",
    page_icon="📈",
    layout="wide"
)

# ─────────────────────────────────────────────
# 2. PATH RESOLUTION
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
# Checking for 'model' (your actual GitHub folder name)
MODELS_DIR = BASE_DIR / "model"

if not MODELS_DIR.exists():
    MODELS_DIR = BASE_DIR / "models" # Fallback

def get_model_path(filename):
    return str(MODELS_DIR / filename)

# ─────────────────────────────────────────────
# 3. FEATURE COLUMN ORDER (33 Total)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "Open", "High", "Low", "Volume",
    "SMA_5", "SMA_10", "SMA_20", "SMA_50",
    "EMA_5", "EMA_10", "EMA_20",
    "BB_width", "BB_upper", "BB_lower",
    "RSI_14", "MACD", "MACD_signal", "MACD_hist",
    "Volatility_5", "Volatility_20",
    "Momentum_5", "Momentum_10",
    "Close_lag_1", "Close_lag_2", "Close_lag_3", "Close_lag_5",
    "Return_lag_1", "Return_lag_2", "Return_lag_3",
    "Volume_ratio", "DayOfWeek", "Month", "Quarter",
]

# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING FUNCTION
# ─────────────────────────────────────────────
def engineer_features(df):
    d = df.copy()
    # Moving averages
    for w in [5, 10, 20, 50, 200]:
        d[f"SMA_{w}"] = d["Close"].rolling(w).mean()
        d[f"EMA_{w}"] = d["Close"].ewm(span=w, adjust=False).mean()
    # Bollinger Bands
    bb_mid = d["Close"].rolling(20).mean()
    bb_std = d["Close"].rolling(20).std()
    d["BB_upper"] = bb_mid + 2 * bb_std
    d["BB_lower"] = bb_mid - 2 * bb_std
    d["BB_width"] = (d["BB_upper"] - d["BB_lower"]) / (bb_mid + 1e-9)
    # RSI
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI_14"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    # MACD
    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACD_signal"]
    # Volatility & Momentum
    ret = d["Close"].pct_change()
    d["Volatility_5"] = ret.rolling(5).std()
    d["Volatility_20"] = ret.rolling(20).std()
    d["Momentum_5"] = d["Close"] - d["Close"].shift(5)
    d["Momentum_10"] = d["Close"] - d["Close"].shift(10)
    # Lags
    for lag in [1, 2, 3, 5]:
        d[f"Close_lag_{lag}"] = d["Close"].shift(lag)
        d[f"Return_lag_{lag}"] = ret.shift(lag)
    # Volume & Calendar
    vol_sma10 = d["Volume"].rolling(10).mean()
    d["Volume_ratio"] = d["Volume"] / (vol_sma10 + 1e-9)
    d["DayOfWeek"] = d.index.dayofweek
    d["Month"] = d.index.month
    d["Quarter"] = d.index.quarter
    return d

# ─────────────────────────────────────────────
# 5. LOADING ARTIFACTS
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    p_path = get_model_path("stock_pipeline_compressed.sav")
    s_path = get_model_path("scaler.sav")
    m_path = get_model_path("pipeline_metadata.sav")
    
    if not os.path.exists(p_path) or not os.path.exists(s_path):
        return None, None, None, f"Missing files in {MODELS_DIR}"
    
    try:
        pipeline = joblib.load(p_path)
        scaler = joblib.load(s_path)
        meta = joblib.load(m_path) if os.path.exists(m_path) else {}
        return pipeline, scaler, meta, None
    except Exception as e:
        return None, None, None, str(e)

# ─────────────────────────────────────────────
# 6. MAIN UI
# ─────────────────────────────────────────────
st.title("📈 StockSight AI")

# Load data
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
predict_btn = st.sidebar.button("RUN PREDICTION")

pipeline, scaler, meta, err = load_artifacts()

if err:
    st.error(f"Error: {err}")
    st.stop()

# Fetch Data
@st.cache_data(ttl=3600)
def fetch_data(symbol):
    data = yf.download(symbol, period="2y", interval="1d", auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

df_raw = fetch_data(ticker)

if not df_raw.empty:
    st.metric("Current Price", f"${df_raw['Close'].iloc[-1]:.2f}")
    
    # Prediction Logic
    if predict_btn:
        df_eng = engineer_features(df_raw)
        # Get latest row with all 33 features
        latest_data = df_eng[FEATURE_COLS].dropna().tail(1)
        
        if not latest_data.empty:
            X_scaled = scaler.transform(latest_data)
            prediction = pipeline.predict(X_scaled)
            st.success(f"### Predicted Next-Day Close: **${prediction[0]:.2f}**")
        else:
            st.warning("Not enough data to calculate all indicators.")
else:
    st.error("Could not fetch data.")
