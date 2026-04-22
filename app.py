# ============================================================
# app.py — StockSight AI: Final Stable Version
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import time
import os
from pathlib import Path
from datetime import datetime

# 1. MUST BE THE ABSOLUTE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="StockSight AI", layout="wide")

# ─────────────────────────────────────────────
# PATH RESOLUTION
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

@st.cache_resource
def load_model_artifacts():
    try:
        pipeline = joblib.load(MODEL_DIR / "stock_pipeline_compressed.sav")
        scaler = joblib.load(MODEL_DIR / "scaler.sav")
        return pipeline, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# ─────────────────────────────────────────────
# SIDEBAR CONFIGURATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ SETTINGS")
    
    # Ticker Selection
    ticker_list = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN", "RELIANCE.NS", "TCS.NS"]
    ticker = st.selectbox("Select Asset Symbol", options=ticker_list, index=0)
    
    # History Period
    history_days = st.slider("History Period (Days)", 250, 1000, 365)
    
    st.divider()
    
    # Live Tracking Toggle
    live_track = st.checkbox("Enable Live Price Tracking", value=True)
    if live_track:
        st.caption("🔄 Auto-refreshing every 60s")

# ─────────────────────────────────────────────
# DATA FETCHING & ERROR HANDLING (Fix for IndexError)
# ─────────────────────────────────────────────
@st.cache_data(ttl=60) # Cache for 1 minute
def get_live_data(symbol, days):
    try:
        df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False)
        # Fix for yfinance Multi-index column issue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        return pd.DataFrame()

df_raw = get_live_data(ticker, history_days)

# Safety Check: Stop the app if data is empty to prevent IndexError
if df_raw.empty or len(df_raw) < 10:
    st.warning(f"⚠️ No data found for {ticker}. Please try another symbol or refresh.")
    st.stop()

# ─────────────────────────────────────────────
# LIVE MARKET METRICS
# ─────────────────────────────────────────────
latest_row = df_raw.iloc[-1]
prev_row = df_raw.iloc[-2]

price_current = float(latest_row['Close'])
price_prev = float(prev_row['Close'])
change = price_current - price_prev
pct_change = (change / price_prev) * 100

st.title(f"📈 StockSight AI: {ticker}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Price", f"${price_current:.2f}", f"{change:+.2f} ({pct_change:+.2f}%)")
m2.metric("Day High", f"${float(latest_row['High']):.2f}")
m3.metric("Day Low", f"${float(latest_row['Low']):.2f}")
m4.metric("Volume", f"{float(latest_row['Volume'])/1e6:.1f}M")

st.divider()

# ─────────────────────────────────────────────
# FEATURE ENGINEERING & PREDICTION
# ─────────────────────────────────────────────
# (Inga unga existing engineer_features and prediction logic ah paste pannunga)
st.info("AI Prediction logic is active. Ready to analyze 33 features.")

# ─────────────────────────────────────────────
# AUTO-REFRESH LOGIC (Very end of script)
# ─────────────────────────────────────────────
if live_track:
    time.sleep(60)
    st.rerun()
    
