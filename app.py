# ============================================================
# app.py — Streamlit Dashboard for Stock Market Prediction
# Description: Loads the saved pipeline and provides live predictions
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
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
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="StockSight AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Retro-Futuristic Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

/* Base theme */
:root {
    --neon-green: #00ff88;
    --neon-blue: #00d4ff;
    --neon-purple: #bf5fff;
    --dark-bg: #0a0e1a;
    --card-bg: #0f1628;
    --border: #1e2d4a;
}

html, body, [class*="css"] {
    background-color: var(--dark-bg) !important;
    color: #c8d8f0 !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--card-bg) !important;
    border-right: 1px solid var(--border) !important;
}

/* Header */
h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--neon-green) !important;
    text-shadow: 0 0 20px rgba(0, 255, 136, 0.4) !important;
}

/* Metric boxes */
[data-testid="metric-container"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--neon-green) !important;
    border-radius: 8px !important;
    padding: 12px !important;
    box-shadow: 0 0 12px rgba(0, 255, 136, 0.1) !important;
}
[data-testid="metric-container"] label {
    color: var(--neon-blue) !important;
    font-size: 11px !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: var(--neon-green) !important;
    font-size: 22px !important;
    font-family: 'Orbitron', sans-serif !important;
}

/* Input widgets */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stDateInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    color: var(--neon-green) !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--neon-green) !important;
    color: var(--neon-green) !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 12px !important;
    letter-spacing: 2px !important;
    transition: all 0.3s !important;
    border-radius: 4px !important;
}
.stButton > button:hover {
    background: rgba(0, 255, 136, 0.1) !important;
    box-shadow: 0 0 20px rgba(0, 255, 136, 0.3) !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--dark-bg); }
::-webkit-scrollbar-thumb { background: var(--neon-green); border-radius: 3px; }

/* Status boxes */
.stSuccess { background: rgba(0, 255, 136, 0.08) !important; border-color: var(--neon-green) !important; }
.stError   { background: rgba(239, 68, 68, 0.08) !important; }
.stInfo    { background: rgba(0, 212, 255, 0.08) !important; border-color: var(--neon-blue) !important; }
.stWarning { background: rgba(245, 158, 11, 0.08) !important; }

/* Code blocks */
code { color: var(--neon-purple) !important; }

/* DataFrame */
.stDataFrame { border: 1px solid var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FEATURE ENGINEERING (mirrors pipeline script)
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer the same technical features as the training pipeline."""
    d = df.copy()

    for window in [5, 10, 20, 50, 200]:
        d[f"SMA_{window}"] = d["Close"].rolling(window).mean()
        d[f"EMA_{window}"] = d["Close"].ewm(span=window, adjust=False).mean()

    d["BB_mid"] = d["Close"].rolling(20).mean()
    d["BB_std"] = d["Close"].rolling(20).std()
    d["BB_upper"] = d["BB_mid"] + 2 * d["BB_std"]
    d["BB_lower"] = d["BB_mid"] - 2 * d["BB_std"]
    d["BB_width"] = (d["BB_upper"] - d["BB_lower"]) / (d["BB_mid"] + 1e-9)

    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    d["RSI_14"] = 100 - (100 / (1 + rs))

    d["Daily_Return"] = d["Close"].pct_change()
    d["Volatility_5"] = d["Daily_Return"].rolling(5).std()
    d["Volatility_20"] = d["Daily_Return"].rolling(20).std()

    d["Momentum_5"] = d["Close"] - d["Close"].shift(5)
    d["Momentum_10"] = d["Close"] - d["Close"].shift(10)

    ema_12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema_12 - ema_26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACD_signal"]

    for lag in [1, 2, 3, 5, 10]:
        d[f"Close_lag_{lag}"] = d["Close"].shift(lag)
        d[f"Return_lag_{lag}"] = d["Daily_Return"].shift(lag)

    d["Volume_SMA_10"] = d["Volume"].rolling(10).mean()
    d["Volume_ratio"] = d["Volume"] / (d["Volume_SMA_10"] + 1e-9)

    d["DayOfWeek"] = d.index.dayofweek
    d["Month"] = d.index.month
    d["Quarter"] = d.index.quarter

    return d


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
# LOAD SAVED ARTIFACTS
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load pipeline, scaler, and metadata from disk."""
    artifacts = {}
    try:
        artifacts["pipeline"] = joblib.load("models/stock_pipeline_compressed.sav")
        artifacts["scaler"] = joblib.load("models/scaler.sav")
        artifacts["metadata"] = joblib.load("models/pipeline_metadata.sav")
        artifacts["loaded"] = True
    except FileNotFoundError:
        artifacts["loaded"] = False
    return artifacts


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ STOCKSIGHT AI")
    st.markdown("---")

    st.markdown("### 🔧 Configuration")
    ticker = st.text_input("Ticker Symbol", value="AAPL", max_chars=10).upper().strip()

    lookback_days = st.slider(
        "Historical Lookback (days)", min_value=90, max_value=730, value=365, step=30
    )

    st.markdown("---")
    st.markdown("### 📊 Chart Settings")
    show_sma = st.checkbox("Show Moving Averages", value=True)
    show_bb = st.checkbox("Show Bollinger Bands", value=True)
    show_volume = st.checkbox("Show Volume", value=True)

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
        st.warning("⚠️ No pipeline found.\nRun `stock_prediction_pipeline.py` first.")

    st.markdown("---")
    st.markdown(
        "<small style='color:#3d5a80'>Built with ❤️ using yfinance, sklearn & Streamlit</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────

st.markdown(
    """
    <div style='text-align:center; padding: 20px 0 10px;'>
        <h1 style='font-size:2.8rem; letter-spacing:6px;'>📈 STOCKSIGHT AI</h1>
        <p style='color:#3d8bcd; font-size:0.85rem; letter-spacing:3px;'>
            MACHINE LEARNING · LIVE MARKET DATA · NEXT-DAY PREDICTION
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()


# ─────────────────────────────────────────────
# DATA FETCHING FUNCTION
# ─────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(ticker: str, days: int) -> pd.DataFrame:
    """Fetch and cache stock data. Refreshes every 5 minutes."""
    end = datetime.today()
    start = end - timedelta(days=days)
    raw = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index)
    raw.sort_index(inplace=True)
    raw.ffill(inplace=True)
    raw.bfill(inplace=True)
    return raw


# ─────────────────────────────────────────────
# MAIN PREDICTION LOGIC
# ─────────────────────────────────────────────

# Auto-run on page load; also runs when button is pressed
with st.spinner(f"⚡ Fetching live data for **{ticker}**..."):
    try:
        df = get_stock_data(ticker, lookback_days + 250)  # extra days for rolling windows
        df_eng = engineer_features(df)
        data_ok = len(df) > 60
    except Exception as e:
        st.error(f"❌ Data fetch error: {e}")
        st.stop()

if not data_ok:
    st.error("❌ Insufficient data. Try a different ticker or increase lookback days.")
    st.stop()

# ── Live Metrics Row ──
latest = df.iloc[-1]
prev = df.iloc[-2]
price_change = latest["Close"] - prev["Close"]
pct_change = (price_change / prev["Close"]) * 100
day_range = f"${latest['Low']:.2f} – ${latest['High']:.2f}"
vol_m = latest["Volume"] / 1_000_000

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("💰 Current Close", f"${latest['Close']:.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
col2.metric("📂 Open", f"${latest['Open']:.2f}")
col3.metric("📊 Day Range", day_range)
col4.metric("📦 Volume", f"{vol_m:.2f}M")
col5.metric("📅 Last Updated", str(df.index[-1].date()))

st.divider()

# ── Price Chart ──
st.markdown("### 📉 PRICE CHART")

df_plot = df_eng.dropna(subset=["SMA_20"]).copy()

fig = make_subplots(
    rows=2 if show_volume else 1,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.75, 0.25] if show_volume else [1.0],
)

# Candlestick
fig.add_trace(
    go.Candlestick(
        x=df_plot.index,
        open=df_plot["Open"], high=df_plot["High"],
        low=df_plot["Low"], close=df_plot["Close"],
        name="OHLC",
        increasing_line_color="#00ff88",
        decreasing_line_color="#ef4444",
    ),
    row=1, col=1,
)

if show_sma:
    for sma, color in [("SMA_20", "#f59e0b"), ("SMA_50", "#3b82f6"), ("SMA_200", "#a855f7")]:
        fig.add_trace(
            go.Scatter(x=df_plot.index, y=df_plot[sma], name=sma,
                       line=dict(color=color, width=1.2)),
            row=1, col=1,
        )

if show_bb:
    fig.add_trace(
        go.Scatter(x=df_plot.index, y=df_plot["BB_upper"], name="BB Upper",
                   line=dict(color="#00d4ff", dash="dash", width=1)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df_plot.index, y=df_plot["BB_lower"], name="BB Lower",
                   line=dict(color="#00d4ff", dash="dash", width=1),
                   fill="tonexty", fillcolor="rgba(0, 212, 255, 0.04)"),
        row=1, col=1,
    )

if show_volume:
    colors = ["#00ff88" if c >= o else "#ef4444"
              for c, o in zip(df_plot["Close"], df_plot["Open"])]
    fig.add_trace(
        go.Bar(x=df_plot.index, y=df_plot["Volume"], name="Volume",
               marker_color=colors, opacity=0.5),
        row=2, col=1,
    )

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0a0e1a",
    plot_bgcolor="#0f1628",
    height=550,
    xaxis_rangeslider_visible=False,
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    margin=dict(l=10, r=10, t=30, b=10),
)
st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# PREDICTION SECTION
# ─────────────────────────────────────────────

st.markdown("### 🤖 NEXT-DAY PREDICTION")

if not artifacts["loaded"]:
    st.warning(
        "⚠️ No saved model found. Please run `stock_prediction_pipeline.py` "
        "to train and save the model first."
    )
else:
    pipeline = artifacts["pipeline"]
    scaler = artifacts["scaler"]

    # Prepare the latest row for prediction
    df_pred_input = df_eng[FEATURE_COLS].dropna()

    if df_pred_input.empty:
        st.error("❌ Not enough data to compute features. Increase lookback days.")
    else:
        latest_features = df_pred_input.iloc[[-1]].values
        latest_features_scaled = scaler.transform(latest_features)
        predicted_price = pipeline.predict(latest_features_scaled)[0]

        current_close = float(df["Close"].iloc[-1])
        prediction_delta = predicted_price - current_close
        prediction_pct = (prediction_delta / current_close) * 100
        direction = "📈 BULLISH" if prediction_delta > 0 else "📉 BEARISH"
        direction_color = "#00ff88" if prediction_delta > 0 else "#ef4444"

        # Prediction box
        st.markdown(
            f"""
            <div style='border:1px solid {direction_color}; border-radius:10px;
                        padding:24px; background:rgba(0,0,0,0.3); margin:10px 0 20px;
                        text-align:center; box-shadow: 0 0 20px {direction_color}33'>
                <div style='font-family:Orbitron,sans-serif; font-size:14px;
                            color:{direction_color}; letter-spacing:4px; margin-bottom:8px;'>
                    PREDICTED NEXT-DAY CLOSE
                </div>
                <div style='font-family:Orbitron,sans-serif; font-size:3rem;
                            color:{direction_color}; text-shadow: 0 0 30px {direction_color};'>
                    ${predicted_price:.2f}
                </div>
                <div style='color:#888; font-size:13px; margin-top:6px;'>
                    {direction} &nbsp;|&nbsp;
                    Change: <span style='color:{direction_color}'>{prediction_delta:+.2f} ({prediction_pct:+.2f}%)</span>
                    &nbsp;vs current close of <span style='color:#fff'>${current_close:.2f}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Historical predictions on recent data ──
        n_hist = min(120, len(df_pred_input))
        hist_features = df_pred_input.iloc[-n_hist:].values
        hist_scaled = scaler.transform(hist_features)
        hist_preds = pipeline.predict(hist_scaled)

        hist_dates = df_pred_input.index[-n_hist:]
        hist_actual = df["Close"].reindex(hist_dates).values

        # Shift predictions by 1 day forward (next-day forecast alignment)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=hist_dates, y=hist_actual,
            name="Actual Close", line=dict(color="#60a5fa", width=2)
        ))
        fig2.add_trace(go.Scatter(
            x=hist_dates, y=hist_preds,
            name="Model Prediction", line=dict(color="#f59e0b", width=2, dash="dot")
        ))
        # Highlight future prediction point
        next_date = hist_dates[-1] + pd.tseries.offsets.BDay(1)
        fig2.add_trace(go.Scatter(
            x=[next_date], y=[predicted_price],
            name="Next-Day Forecast",
            mode="markers",
            marker=dict(color=direction_color, size=14, symbol="diamond",
                        line=dict(color="white", width=1.5))
        ))
        fig2.update_layout(
            title="Recent Actual vs Predicted Prices",
            template="plotly_dark",
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#0f1628",
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Model metrics from saved metadata ──
        st.markdown("### 🏆 MODEL PERFORMANCE (Test Set)")
        meta = artifacts["metadata"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", f"${meta.get('test_mae', 'N/A')}")
        m2.metric("RMSE", f"${meta.get('test_rmse', 'N/A')}")
        m3.metric("R² Score", f"{meta.get('test_r2', 'N/A')}")
        m4.metric("MAPE", f"{meta.get('test_mape', 'N/A')}%")


# ─────────────────────────────────────────────
# RSI & MACD INDICATORS
# ─────────────────────────────────────────────

with st.expander("📡 TECHNICAL INDICATORS", expanded=False):
    ind_col1, ind_col2 = st.columns(2)

    with ind_col1:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df_eng.index, y=df_eng["RSI_14"],
            name="RSI 14", line=dict(color="#f59e0b", width=1.5)
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef4444")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#00ff88")
        fig_rsi.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.05)", line_width=0)
        fig_rsi.add_hrect(y0=0, y1=30, fillcolor="rgba(0,255,136,0.05)", line_width=0)
        fig_rsi.update_layout(
            title="RSI (14)", template="plotly_dark",
            paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1628",
            height=280, yaxis=dict(range=[0, 100]),
            margin=dict(l=5, r=5, t=40, b=5),
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

    with ind_col2:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(
            x=df_eng.index, y=df_eng["MACD"],
            name="MACD", line=dict(color="#00d4ff", width=1.5)
        ))
        fig_macd.add_trace(go.Scatter(
            x=df_eng.index, y=df_eng["MACD_signal"],
            name="Signal", line=dict(color="#f59e0b", width=1.2)
        ))
        colors = ["#00ff88" if v >= 0 else "#ef4444"
                  for v in df_eng["MACD_hist"].fillna(0)]
        fig_macd.add_trace(go.Bar(
            x=df_eng.index, y=df_eng["MACD_hist"],
            name="Histogram", marker_color=colors, opacity=0.6
        ))
        fig_macd.update_layout(
            title="MACD", template="plotly_dark",
            paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1628",
            height=280, margin=dict(l=5, r=5, t=40, b=5),
        )
        st.plotly_chart(fig_macd, use_container_width=True)


# ─────────────────────────────────────────────
# RECENT DATA TABLE
# ─────────────────────────────────────────────

with st.expander("📋 RECENT MARKET DATA", expanded=False):
    display_df = df[["Open", "High", "Low", "Close", "Volume"]].tail(20).copy()
    display_df.index = display_df.index.strftime("%Y-%m-%d")
    display_df = display_df.round(2)
    display_df["Volume"] = display_df["Volume"].apply(lambda x: f"{x/1e6:.2f}M")
    st.dataframe(display_df, use_container_width=True)
