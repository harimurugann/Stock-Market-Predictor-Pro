# ============================================================
# app.py — StockSight AI · FINAL STABLE VERSION
# Fixes: Folder path mismatch (model vs models) & Feature Shape
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
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path

# ─────────────────────────────────────────────
# PATH RESOLUTION (FIXED FOR YOUR REPO)
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# Checking for 'model' (your GitHub folder name)
MODELS_DIR = BASE_DIR / "model" 

# Fallback in case you rename it to 'models' later
if not MODELS_DIR.exists():
    MODELS_DIR = BASE_DIR / "models"

def get_model_path(filename: str) -> str:
    return str(MODELS_DIR / filename)

# ─────────────────────────────────────────────
# FEATURE COLUMN ORDER (33 FEATURES)
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

# Total: 33 ✓
assert len(FEATURE_COLS) == 33

# ... (Keeping your existing CSS and Page Config from the prompt) ...
st.set_page_config(page_title="StockSight AI", layout="wide")

# (Paste the CSS block here if you want the neon green theme back)

# ═══════════════════════════════════════════════════════════════
# CORE: FEATURE ENGINEERING (Ensuring all 33 are calculated)
# ═══════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
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
    d["Daily_Return"] = ret
    d["Volatility_5"] = ret.rolling(5).std()
    d["Volatility_20"] = ret.rolling(20).std()
    d["Momentum_5"] = d["Close"] - d["Close"].shift(5)
    d["Momentum_10"] = d["Close"] - d["Close"].shift(10)
    # Lags
    for lag in [1, 2, 3, 5, 10]:
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
# LOAD MODEL ARTIFACTS (FIXED PATH)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    out = {"loaded": False}
    files = {
        "pipeline": get_model_path("stock_pipeline_compressed.sav"),
        "scaler":   get_model_path("scaler.sav"),
        "metadata": get_model_path("pipeline_metadata.sav"),
    }
    
    # Check if files exist
    for k, v in files.items():
        if k != "metadata" and not os.path.exists(v):
            out["error"] = f"File missing: {v}"
            return out

    try:
        out["pipeline"] = joblib.load(files["pipeline"])
        out["scaler"]   = joblib.load(files["scaler"])
        out["metadata"] = joblib.load(files["metadata"]) if os.path.exists(files["metadata"]) else {}
        out["loaded"]   = True
    except Exception as e:
        out["error"] = f"Load error: {e}"
    return out

# ... (Continue with your Sidebar, Charting, and Prediction logic as provided in your prompt) ...
# Ensure you use 'get_stock_data(ticker)' and 'engineer_features(df_raw)'
