# ============================================================
# app.py — StockSight AI · Fixed Version
# Fixes: (1) robust model path resolution via __file__
#        (2) yfinance retry with period='2y' fallback
#        (3) SMA-200 warmup guard before feature engineering
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

# ─────────────────────────────────────────────
# PATH RESOLUTION  ← FIX #1
# Always resolve paths relative to THIS script's location,
# so the app works on Streamlit Cloud, local, and any OS.
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")   # works even if cwd is different

def model_path(filename: str) -> str:
    """Return the absolute path to a file inside the models/ folder."""
    return os.path.join(MODELS_DIR, filename)

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="StockSight AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Retro-Futuristic Dark Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

:root {
    --neon-green:  #00ff88;
    --neon-blue:   #00d4ff;
    --neon-purple: #bf5fff;
    --dark-bg:     #0a0e1a;
    --card-bg:     #0f1628;
    --border:      #1e2d4a;
}

html, body, [class*="css"] {
    background-color: var(--dark-bg)  !important;
    color: #c8d8f0                    !important;
    font-family: 'Share Tech Mono', monospace !important;
}
section[data-testid="stSidebar"] {
    background: var(--card-bg) !important;
    border-right: 1px solid var(--border) !important;
}
h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--neon-green)           !important;
    text-shadow: 0 0 20px rgba(0,255,136,.4) !important;
}
[data-testid="metric-container"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--neon-green) !important;
    border-radius: 8px !important;
    padding: 12px   !important;
    box-shadow: 0 0 12px rgba(0,255,136,.1) !important;
}
[data-testid="metric-container"] label {
    color: var(--neon-blue) !important;
    font-size: 11px         !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: var(--neon-green)  !important;
    font-size: 22px           !important;
    font-family: 'Orbitron', sans-serif !important;
}
.stButton > button {
    background: transparent                     !important;
    border: 1px solid var(--neon-green)         !important;
    color: var(--neon-green)                    !important;
    font-family: 'Orbitron', sans-serif         !important;
    font-size: 12px; letter-spacing: 2px        !important;
    transition: all .3s                         !important;
    border-radius: 4px                          !important;
}
.stButton > button:hover {
    background: rgba(0,255,136,.1)              !important;
    box-shadow: 0 0 20px rgba(0,255,136,.3)     !important;
}
hr { border-color: var(--border) !important; }
::-webkit-scrollbar       { width: 6px; }
::-webkit-scrollbar-track { background: var(--dark-bg); }
::-webkit-scrollbar-thumb { background: var(--neon-green); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FEATURE COLUMNS (must match training script)
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
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for w in [5, 10, 20, 50, 200]:
        d[f"SMA_{w}"] = d["Close"].rolling(w).mean()
        d[f"EMA_{w}"] = d["Close"].ewm(span=w, adjust=False).mean()

    d["BB_mid"]   = d["Close"].rolling(20).mean()
    d["BB_std"]   = d["Close"].rolling(20).std()
    d["BB_upper"] = d["BB_mid"] + 2 * d["BB_std"]
    d["BB_lower"] = d["BB_mid"] - 2 * d["BB_std"]
    d["BB_width"] = (d["BB_upper"] - d["BB_lower"]) / (d["BB_mid"] + 1e-9)

    delta = d["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI_14"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    d["Daily_Return"]  = d["Close"].pct_change()
    d["Volatility_5"]  = d["Daily_Return"].rolling(5).std()
    d["Volatility_20"] = d["Daily_Return"].rolling(20).std()
    d["Momentum_5"]    = d["Close"] - d["Close"].shift(5)
    d["Momentum_10"]   = d["Close"] - d["Close"].shift(10)

    ema12, ema26 = d["Close"].ewm(span=12,adjust=False).mean(), d["Close"].ewm(span=26,adjust=False).mean()
    d["MACD"]        = ema12 - ema26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"]   = d["MACD"] - d["MACD_signal"]

    for lag in [1, 2, 3, 5, 10]:
        d[f"Close_lag_{lag}"]  = d["Close"].shift(lag)
        d[f"Return_lag_{lag}"] = d["Daily_Return"].shift(lag)

    d["Volume_SMA_10"] = d["Volume"].rolling(10).mean()
    d["Volume_ratio"]  = d["Volume"] / (d["Volume_SMA_10"] + 1e-9)
    d["DayOfWeek"]     = d.index.dayofweek
    d["Month"]         = d.index.month
    d["Quarter"]       = d.index.quarter
    return d

# ─────────────────────────────────────────────
# LOAD MODEL ARTIFACTS  ← FIX #1 (path)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    out = {"loaded": False}
    pipeline_file = model_path("stock_pipeline_compressed.sav")
    scaler_file   = model_path("scaler.sav")
    meta_file     = model_path("pipeline_metadata.sav")

    # Debug info shown only when files are missing
    if not os.path.exists(pipeline_file):
        out["error"] = (
            f"Pipeline not found at: `{pipeline_file}`\n\n"
            "Make sure your repo has a `models/` folder containing "
            "`stock_pipeline_compressed.sav`, `scaler.sav`, and "
            "`pipeline_metadata.sav`, and that they are **not** listed in `.gitignore`."
        )
        return out

    try:
        out["pipeline"] = joblib.load(pipeline_file)
        out["scaler"]   = joblib.load(scaler_file)
        out["metadata"] = joblib.load(meta_file) if os.path.exists(meta_file) else {}
        out["loaded"]   = True
    except Exception as e:
        out["error"] = f"Failed to load model files: {e}"
    return out

# ─────────────────────────────────────────────
# DATA FETCHING WITH RETRY  ← FIX #2
# Strategy: always use period='2y' so yfinance returns ~500 trading
# days — enough for SMA-200 warmup + feature lags.
# Falls back to a longer period on failure, with 3 retries.
# ─────────────────────────────────────────────
MIN_ROWS_REQUIRED = 250   # need 200 for SMA-200 + 50 extra for other lags

@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(ticker: str) -> pd.DataFrame:
    """
    Fetch 2 years of daily OHLCV data.
    Retries up to 3 times with exponential back-off.
    Raises a descriptive RuntimeError on failure.
    """
    periods_to_try = ["2y", "3y", "5y"]   # escalating fallback
    last_error = None

    for attempt, period in enumerate(periods_to_try, start=1):
        try:
            raw = yf.download(
                ticker,
                period=period,
                interval="1d",
                progress=False,
                auto_adjust=True,
                timeout=15,        # seconds; avoids silent hangs
            )

            # Flatten multi-level columns that yfinance sometimes returns
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            if raw.empty:
                raise ValueError(f"yfinance returned an empty DataFrame for '{ticker}'.")

            raw.index = pd.to_datetime(raw.index)
            raw.sort_index(inplace=True)
            raw.ffill(inplace=True)
            raw.bfill(inplace=True)

            if len(raw) < MIN_ROWS_REQUIRED:
                raise ValueError(
                    f"Only {len(raw)} rows returned (need ≥{MIN_ROWS_REQUIRED}). "
                    "Trying a longer period…"
                )

            return raw   # success

        except Exception as e:
            last_error = e
            if attempt < len(periods_to_try):
                time.sleep(2 ** attempt)   # wait 2s, 4s before next retry
            continue

    # All attempts failed
    raise RuntimeError(
        f"Could not fetch data for '{ticker}' after {len(periods_to_try)} attempts.\n"
        f"Last error: {last_error}\n\n"
        "Possible causes:\n"
        "• Invalid ticker symbol\n"
        "• yfinance API rate-limit or outage (try again in a minute)\n"
        "• Network restrictions on the hosting environment"
    )

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ STOCKSIGHT AI")
    st.markdown("---")
    st.markdown("### 🔧 Configuration")

    ticker = st.text_input("Ticker Symbol", value="AAPL", max_chars=10).upper().strip()

    st.markdown("---")
    st.markdown("### 📊 Chart Settings")
    show_sma    = st.checkbox("Show Moving Averages", value=True)
    show_bb     = st.checkbox("Show Bollinger Bands",  value=True)
    show_volume = st.checkbox("Show Volume",            value=True)

    st.markdown("---")
    predict_btn = st.button("🚀 RUN PREDICTION", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Model Info")
    artifacts = load_artifacts()

    if artifacts["loaded"]:
        meta = artifacts["metadata"]
        st.success("✅ Pipeline Loaded")
        st.markdown(f"`Trained on:` **{meta.get('ticker', 'N/A')}**")
        st.markdown(f"`Train cutoff:` **{meta.get('train_end_date', 'N/A')}**")
        st.markdown(f"`R²:` **{meta.get('test_r2', 'N/A')}**")
        st.markdown(f"`MAE:` **${meta.get('test_mae', 'N/A')}**")
    else:
        err_msg = artifacts.get("error", "Unknown error loading pipeline.")
        st.warning(f"⚠️ No pipeline found.\n\n{err_msg}")

    st.markdown("---")
    st.markdown(
        "<small style='color:#3d5a80'>Built with ❤️ using yfinance, sklearn & Streamlit</small>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:20px 0 10px'>
  <h1 style='font-size:2.8rem; letter-spacing:6px'>📈 STOCKSIGHT AI</h1>
  <p style='color:#3d8bcd; font-size:.85rem; letter-spacing:3px'>
    MACHINE LEARNING · LIVE MARKET DATA · NEXT-DAY PREDICTION
  </p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────
# DATA LOADING  ← FIX #2 + #3 (try/except)
# ─────────────────────────────────────────────
with st.spinner(f"⚡ Fetching 2 years of live data for **{ticker}**..."):
    try:
        df = get_stock_data(ticker)
    except RuntimeError as e:
        st.error(f"❌ **Data fetch failed.**\n\n{e}")
        st.info(
            "💡 **Tips to fix this:**\n"
            "- Double-check the ticker symbol (e.g. `AAPL`, `TSLA`, `RELIANCE.NS`)\n"
            "- Wait 60 seconds and try again (API rate limit)\n"
            "- If on Streamlit Cloud, the outbound network may be restricted"
        )
        st.stop()

# Engineer features
df_eng = engineer_features(df)

# Guard: ensure enough clean rows after dropping NaN from rolling windows
df_model_ready = df_eng[FEATURE_COLS].dropna()
if len(df_model_ready) < 60:
    st.error(
        f"❌ After feature engineering, only **{len(df_model_ready)}** usable rows remain "
        f"(need ≥ 60). The ticker `{ticker}` may not have enough trading history."
    )
    st.stop()

# ─────────────────────────────────────────────
# LIVE METRICS ROW
# ─────────────────────────────────────────────
latest      = df.iloc[-1]
prev        = df.iloc[-2]
price_chg   = float(latest["Close"]) - float(prev["Close"])
pct_chg     = price_chg / float(prev["Close"]) * 100
day_range   = f"${float(latest['Low']):.2f} – ${float(latest['High']):.2f}"
vol_m       = float(latest["Volume"]) / 1_000_000

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("💰 Close",        f"${float(latest['Close']):.2f}", f"{price_chg:+.2f} ({pct_chg:+.2f}%)")
c2.metric("📂 Open",         f"${float(latest['Open']):.2f}")
c3.metric("📊 Day Range",    day_range)
c4.metric("📦 Volume",       f"{vol_m:.2f}M")
c5.metric("📅 Last Updated", str(df.index[-1].date()))

st.divider()

# ─────────────────────────────────────────────
# PRICE CHART
# ─────────────────────────────────────────────
st.markdown("### 📉 PRICE CHART")

df_plot = df_eng.dropna(subset=["SMA_20"]).copy()
fig = make_subplots(
    rows=2 if show_volume else 1, cols=1,
    shared_xaxes=True, vertical_spacing=.03,
    row_heights=[.75, .25] if show_volume else [1.0],
)
fig.add_trace(go.Candlestick(
    x=df_plot.index,
    open=df_plot["Open"], high=df_plot["High"],
    low=df_plot["Low"],   close=df_plot["Close"],
    name="OHLC",
    increasing_line_color="#00ff88",
    decreasing_line_color="#ef4444",
), row=1, col=1)

if show_sma:
    for col, color in [("SMA_20","#f59e0b"),("SMA_50","#3b82f6"),("SMA_200","#a855f7")]:
        if col in df_plot.columns:
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[col],
                                     name=col, line=dict(color=color, width=1.2)), row=1, col=1)

if show_bb and "BB_upper" in df_plot.columns:
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["BB_upper"], name="BB Upper",
                             line=dict(color="#00d4ff", dash="dash", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["BB_lower"], name="BB Lower",
                             line=dict(color="#00d4ff", dash="dash", width=1),
                             fill="tonexty", fillcolor="rgba(0,212,255,.04)"), row=1, col=1)

if show_volume:
    bar_colors = ["#00ff88" if c >= o else "#ef4444"
                  for c, o in zip(df_plot["Close"], df_plot["Open"])]
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot["Volume"],
                         name="Volume", marker_color=bar_colors, opacity=.5), row=2, col=1)

fig.update_layout(
    template="plotly_dark", paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1628",
    height=550, xaxis_rangeslider_visible=False,
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    margin=dict(l=10, r=10, t=30, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PREDICTION SECTION
# ─────────────────────────────────────────────
st.markdown("### 🤖 NEXT-DAY PREDICTION")

if not artifacts["loaded"]:
    err_detail = artifacts.get("error", "")
    st.warning(
        "⚠️ **No saved model found.**\n\n"
        "Run `stock_prediction_pipeline.py` locally to train & save the model, "
        "then commit the `models/` folder to your repo.\n\n"
        + (f"*Debug:* {err_detail}" if err_detail else "")
    )
else:
    pipeline = artifacts["pipeline"]
    scaler   = artifacts["scaler"]

    latest_feat = df_model_ready.iloc[[-1]].values
    try:
        latest_scaled    = scaler.transform(latest_feat)
        predicted_price  = float(pipeline.predict(latest_scaled)[0])
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    current_close  = float(df["Close"].iloc[-1])
    delta          = predicted_price - current_close
    pct            = delta / current_close * 100
    direction      = "📈 BULLISH" if delta > 0 else "📉 BEARISH"
    d_color        = "#00ff88" if delta > 0 else "#ef4444"

    st.markdown(f"""
    <div style='border:1px solid {d_color}; border-radius:10px; padding:24px;
                background:rgba(0,0,0,.3); margin:10px 0 20px; text-align:center;
                box-shadow:0 0 20px {d_color}33'>
        <div style='font-family:Orbitron,sans-serif; font-size:14px;
                    color:{d_color}; letter-spacing:4px; margin-bottom:8px;'>
            PREDICTED NEXT-DAY CLOSE
        </div>
        <div style='font-family:Orbitron,sans-serif; font-size:3rem;
                    color:{d_color}; text-shadow:0 0 30px {d_color};'>
            ${predicted_price:.2f}
        </div>
        <div style='color:#888; font-size:13px; margin-top:6px;'>
            {direction} &nbsp;|&nbsp;
            Change: <span style='color:{d_color}'>{delta:+.2f} ({pct:+.2f}%)</span>
            &nbsp;vs current close of <span style='color:#fff'>${current_close:.2f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Historical prediction overlay (last 120 rows)
    n_hist   = min(120, len(df_model_ready))
    h_feat   = df_model_ready.iloc[-n_hist:].values
    h_scaled = scaler.transform(h_feat)
    h_pred   = pipeline.predict(h_scaled)
    h_dates  = df_model_ready.index[-n_hist:]
    h_actual = df["Close"].reindex(h_dates).values.flatten()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=h_dates, y=h_actual,
                              name="Actual Close", line=dict(color="#60a5fa", width=2)))
    fig2.add_trace(go.Scatter(x=h_dates, y=h_pred,
                              name="Model Prediction",
                              line=dict(color="#f59e0b", width=2, dash="dot")))
    next_date = h_dates[-1] + pd.tseries.offsets.BDay(1)
    fig2.add_trace(go.Scatter(x=[next_date], y=[predicted_price],
                              name="Next-Day Forecast", mode="markers",
                              marker=dict(color=d_color, size=14, symbol="diamond",
                                          line=dict(color="white", width=1.5))))
    fig2.update_layout(
        title="Recent Actual vs Predicted Prices",
        template="plotly_dark", paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1628",
        height=400, margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Model metrics
    st.markdown("### 🏆 MODEL PERFORMANCE (Test Set)")
    meta = artifacts["metadata"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE",      f"${meta.get('test_mae',  'N/A')}")
    m2.metric("RMSE",     f"${meta.get('test_rmse', 'N/A')}")
    m3.metric("R² Score", f"{meta.get('test_r2',   'N/A')}")
    m4.metric("MAPE",     f"{meta.get('test_mape', 'N/A')}%")

# ─────────────────────────────────────────────
# TECHNICAL INDICATORS EXPANDER
# ─────────────────────────────────────────────
with st.expander("📡 TECHNICAL INDICATORS", expanded=False):
    ic1, ic2 = st.columns(2)
    with ic1:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df_eng.index, y=df_eng["RSI_14"],
                                     name="RSI 14", line=dict(color="#f59e0b", width=1.5)))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef4444")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#00ff88")
        fig_rsi.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,.05)",  line_width=0)
        fig_rsi.add_hrect(y0=0,  y1=30,  fillcolor="rgba(0,255,136,.05)", line_width=0)
        fig_rsi.update_layout(title="RSI (14)", template="plotly_dark",
                               paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1628",
                               height=280, yaxis=dict(range=[0,100]),
                               margin=dict(l=5,r=5,t=40,b=5))
        st.plotly_chart(fig_rsi, use_container_width=True)
    with ic2:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df_eng.index, y=df_eng["MACD"],
                                      name="MACD", line=dict(color="#00d4ff", width=1.5)))
        fig_macd.add_trace(go.Scatter(x=df_eng.index, y=df_eng["MACD_signal"],
                                      name="Signal", line=dict(color="#f59e0b", width=1.2)))
        macd_colors = ["#00ff88" if v >= 0 else "#ef4444"
                       for v in df_eng["MACD_hist"].fillna(0)]
        fig_macd.add_trace(go.Bar(x=df_eng.index, y=df_eng["MACD_hist"],
                                  name="Histogram", marker_color=macd_colors, opacity=.6))
        fig_macd.update_layout(title="MACD", template="plotly_dark",
                                paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1628",
                                height=280, margin=dict(l=5,r=5,t=40,b=5))
        st.plotly_chart(fig_macd, use_container_width=True)

# ─────────────────────────────────────────────
# RECENT DATA TABLE
# ─────────────────────────────────────────────
with st.expander("📋 RECENT MARKET DATA", expanded=False):
    disp = df[["Open","High","Low","Close","Volume"]].tail(20).copy()
    disp.index = disp.index.strftime("%Y-%m-%d")
    disp = disp.round(2)
    disp["Volume"] = disp["Volume"].apply(lambda x: f"{float(x)/1e6:.2f}M")
    st.dataframe(disp, use_container_width=True)
