# ============================================================
# app.py — StockSight AI
# Fixes: blank screen, model/ path (singular), 33 features
# ============================================================
#
# RULE: Nothing runs before st.set_page_config().
# Even import errors must be caught AFTER page config.
# ============================================================

import streamlit as st  # must be first import

# ── PAGE CONFIG — must be the very first st.* call ──────────
st.set_page_config(
    page_title="StockSight AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── All other imports come AFTER set_page_config ────────────
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Wrap every heavy import in try/except so a missing package
# shows an error instead of a blank screen
try:
    import yfinance as yf
except ImportError:
    st.error("❌ `yfinance` not installed. Add it to requirements.txt")
    st.stop()

try:
    import joblib
except ImportError:
    st.error("❌ `joblib` not installed. Add it to requirements.txt")
    st.stop()

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
except ImportError:
    st.error("❌ `plotly` not installed. Add it to requirements.txt")
    st.stop()

# ════════════════════════════════════════════════════════════
# PATH RESOLUTION
# Streamlit Cloud clones your repo to:
#   /mount/src/<repo-name>/
# __file__ will be something like:
#   /mount/src/stock-market-predictor-pro/app.py
# So Path(__file__).parent / "model" gives the correct folder
# regardless of whether you're running locally or on Cloud.
# ════════════════════════════════════════════════════════════
BASE_DIR   = Path(__file__).resolve().parent   # folder containing app.py
MODEL_DIR  = BASE_DIR / "model"                # singular — matches your GitHub repo

def mpath(filename: str) -> Path:
    """Return absolute path to a file inside the model/ folder."""
    return MODEL_DIR / filename

# Write a visible debug line immediately so we know the app started
st.markdown(
    f"<small style='color:#3d5a80'>🔍 Looking for models in: `{MODEL_DIR}`</small>",
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════
# FEATURE COLUMN ORDER
# Must exactly match the order passed to scaler.fit_transform()
# in stock_prediction_pipeline.py.
# 4 + 4 + 3 + 3 + 4 + 2 + 2 + 4 + 3 + 4 = 33
# ════════════════════════════════════════════════════════════
FEATURE_COLS = [
    # ── Raw price/volume (4) ──────────────────────────────────
    "Open", "High", "Low", "Volume",
    # ── Simple Moving Averages (4) ───────────────────────────
    "SMA_5", "SMA_10", "SMA_20", "SMA_50",
    # ── Exponential Moving Averages (3) ──────────────────────
    "EMA_5", "EMA_10", "EMA_20",
    # ── Bollinger Bands (3) ───────────────────────────────────
    "BB_width", "BB_upper", "BB_lower",
    # ── Oscillators (4) ───────────────────────────────────────
    "RSI_14", "MACD", "MACD_signal", "MACD_hist",
    # ── Volatility (2) ────────────────────────────────────────
    "Volatility_5", "Volatility_20",
    # ── Price momentum (2) ────────────────────────────────────
    "Momentum_5", "Momentum_10",
    # ── Close lag features (4) ────────────────────────────────
    "Close_lag_1", "Close_lag_2", "Close_lag_3", "Close_lag_5",
    # ── Return lag features (3) ───────────────────────────────
    "Return_lag_1", "Return_lag_2", "Return_lag_3",
    # ── Volume + calendar (4) ─────────────────────────────────
    "Volume_ratio", "DayOfWeek", "Month", "Quarter",
]

if len(FEATURE_COLS) != 33:
    st.error(f"❌ FEATURE_COLS has {len(FEATURE_COLS)} entries — expected 33. Fix the list.")
    st.stop()

# ════════════════════════════════════════════════════════════
# CSS — Neon Green Dark Theme
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

:root {
    --neon-green:  #00ff88;
    --neon-blue:   #00d4ff;
    --dark-bg:     #0a0e1a;
    --card-bg:     #0f1628;
    --border:      #1e2d4a;
}
html, body, [class*="css"] {
    background-color: var(--dark-bg) !important;
    color: #c8d8f0 !important;
    font-family: 'Share Tech Mono', monospace !important;
}
section[data-testid="stSidebar"] {
    background: var(--card-bg) !important;
    border-right: 1px solid var(--border) !important;
}
h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--neon-green) !important;
    text-shadow: 0 0 20px rgba(0,255,136,.4) !important;
}
[data-testid="metric-container"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--neon-green) !important;
    border-radius: 8px !important;
    padding: 12px !important;
    box-shadow: 0 0 12px rgba(0,255,136,.1) !important;
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
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--neon-green) !important;
    color: var(--neon-green) !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 12px !important;
    letter-spacing: 2px !important;
    transition: all .3s !important;
    border-radius: 4px !important;
}
.stButton > button:hover {
    background: rgba(0,255,136,.1) !important;
    box-shadow: 0 0 20px rgba(0,255,136,.3) !important;
}
hr { border-color: var(--border) !important; }
::-webkit-scrollbar       { width: 6px; }
::-webkit-scrollbar-track { background: var(--dark-bg); }
::-webkit-scrollbar-thumb { background: var(--neon-green); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# Produces all 33 columns in FEATURE_COLS order.
# Input DataFrame needs >= 215 rows (200 for SMA-200 + buffer).
# ════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Moving averages
    for w in [5, 10, 20, 50, 200]:
        d[f"SMA_{w}"] = d["Close"].rolling(w).mean()
        d[f"EMA_{w}"] = d["Close"].ewm(span=w, adjust=False).mean()

    # Bollinger Bands
    bb_mid        = d["Close"].rolling(20).mean()
    bb_std        = d["Close"].rolling(20).std()
    d["BB_upper"] = bb_mid + 2 * bb_std
    d["BB_lower"] = bb_mid - 2 * bb_std
    d["BB_width"] = (d["BB_upper"] - d["BB_lower"]) / (bb_mid + 1e-9)

    # RSI 14-day
    delta       = d["Close"].diff()
    gain        = delta.clip(lower=0).rolling(14).mean()
    loss        = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI_14"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # MACD
    ema12            = d["Close"].ewm(span=12, adjust=False).mean()
    ema26            = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"]        = ema12 - ema26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"]   = d["MACD"] - d["MACD_signal"]

    # Returns & volatility
    ret                = d["Close"].pct_change()
    d["Daily_Return"]  = ret
    d["Volatility_5"]  = ret.rolling(5).std()
    d["Volatility_20"] = ret.rolling(20).std()

    # Momentum
    d["Momentum_5"]  = d["Close"] - d["Close"].shift(5)
    d["Momentum_10"] = d["Close"] - d["Close"].shift(10)

    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        d[f"Close_lag_{lag}"]  = d["Close"].shift(lag)
        d[f"Return_lag_{lag}"] = ret.shift(lag)

    # Volume ratio
    d["Volume_ratio"] = d["Volume"] / (d["Volume"].rolling(10).mean() + 1e-9)

    # Calendar
    d["DayOfWeek"] = d.index.dayofweek
    d["Month"]     = d.index.month
    d["Quarter"]   = d.index.quarter

    return d


def get_latest_features(df_engineered: pd.DataFrame) -> np.ndarray:
    """
    Return shape (1, 33) from the last complete (non-NaN) row.
    Raises a clear ValueError if anything is wrong.
    """
    missing = [c for c in FEATURE_COLS if c not in df_engineered.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} feature columns after engineering: {missing}")

    clean = df_engineered[FEATURE_COLS].dropna()
    if clean.empty:
        raise ValueError(
            "Zero complete rows after dropping NaN — "
            "not enough historical data to compute all rolling indicators. "
            "Try a longer period or a different ticker."
        )

    row = clean.iloc[[-1]].values
    if row.shape != (1, 33):
        raise ValueError(f"Expected shape (1, 33), got {row.shape}")
    return row


# ════════════════════════════════════════════════════════════
# MODEL LOADING — wrapped so errors show instead of blank screen
# ════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_artifacts():
    result = {"loaded": False, "error": None}

    pipeline_file = mpath("stock_pipeline_compressed.sav")
    scaler_file   = mpath("scaler.sav")
    meta_file     = mpath("pipeline_metadata.sav")

    # Check files exist and report exact missing paths
    missing_files = [str(f) for f in [pipeline_file, scaler_file] if not f.exists()]
    if missing_files:
        result["error"] = (
            "Model file(s) not found:\n" +
            "\n".join(f"  • `{f}`" for f in missing_files) +
            "\n\nFix checklist:\n"
            "1. Your repo folder must be named `model` (singular, lowercase)\n"
            "2. The `.sav` files must be committed (not in `.gitignore`)\n"
            "3. Re-run `stock_prediction_pipeline.py` and push the output"
        )
        return result

    try:
        result["pipeline"] = joblib.load(pipeline_file)
        result["scaler"]   = joblib.load(scaler_file)
        result["metadata"] = joblib.load(meta_file) if meta_file.exists() else {}
        result["loaded"]   = True

        # Verify scaler expects exactly 33 features
        n = result["scaler"].n_features_in_
        if n != 33:
            result["loaded"] = False
            result["error"]  = (
                f"Loaded scaler expects **{n}** features, but this app sends **33**.\n\n"
                "Re-train `stock_prediction_pipeline.py` so the pipeline "
                "and the feature list in `app.py` are in sync."
            )
    except Exception:
        result["error"] = (
            "Exception while loading model files:\n\n"
            f"```\n{traceback.format_exc()}\n```"
        )

    return result


# ════════════════════════════════════════════════════════════
# DATA FETCHING — period='2y' guarantees enough warmup rows
# ════════════════════════════════════════════════════════════
MIN_ROWS = 215  # 200 for SMA-200 + 15 buffer for lag features

@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(ticker: str) -> pd.DataFrame:
    periods  = ["2y", "3y", "5y"]
    last_err = None

    for attempt, period in enumerate(periods, 1):
        try:
            raw = yf.download(
                ticker,
                period=period,
                interval="1d",
                progress=False,
                auto_adjust=True,
                timeout=20,
            )

            # Flatten multi-level columns (yfinance >= 0.2.x)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            if raw.empty:
                raise ValueError(f"yfinance returned empty data for '{ticker}'")

            raw.index = pd.to_datetime(raw.index)
            raw.sort_index(inplace=True)
            raw.ffill(inplace=True)
            raw.bfill(inplace=True)

            if len(raw) < MIN_ROWS:
                raise ValueError(
                    f"Only {len(raw)} rows returned (need ≥{MIN_ROWS}). "
                    "Trying longer period…"
                )

            return raw

        except Exception as e:
            last_err = e
            if attempt < len(periods):
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Failed to fetch data for '{ticker}' after {len(periods)} attempts.\n"
        f"Last error: {last_err}\n\n"
        "Possible causes: invalid ticker · API rate limit · network restriction"
    )


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ STOCKSIGHT AI")
    st.divider()

    st.markdown("### 🔧 Configuration")
    ticker = st.text_input("Ticker Symbol", value="AAPL", max_chars=10).upper().strip()

    st.divider()
    st.markdown("### 📊 Chart Settings")
    show_sma    = st.checkbox("Show Moving Averages", value=True)
    show_bb     = st.checkbox("Show Bollinger Bands",  value=True)
    show_volume = st.checkbox("Show Volume",            value=True)

    st.divider()
    predict_btn = st.button("🚀 RUN PREDICTION", use_container_width=True)

    st.divider()
    st.markdown("### 📋 Model Info")
    artifacts = load_artifacts()

    if artifacts["loaded"]:
        meta = artifacts["metadata"]
        st.success("✅ Pipeline Loaded")
        st.markdown(f"`Path:` `{MODEL_DIR.name}/`")
        st.markdown(f"`Features:` **{artifacts['scaler'].n_features_in_}**")
        st.markdown(f"`Trained on:` **{meta.get('ticker', 'N/A')}**")
        st.markdown(f"`Cutoff:` **{meta.get('train_end_date', 'N/A')}**")
        st.markdown(f"`R²:` **{meta.get('test_r2', 'N/A')}**")
        st.markdown(f"`MAE:` **${meta.get('test_mae', 'N/A')}**")
    else:
        st.error("❌ Pipeline not loaded")
        if artifacts.get("error"):
            st.markdown(artifacts["error"])

    st.divider()
    st.markdown(
        "<small style='color:#3d5a80'>yfinance · sklearn · Streamlit</small>",
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════
# MAIN HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center; padding:20px 0 10px'>
  <h1 style='font-size:2.8rem; letter-spacing:6px'>📈 STOCKSIGHT AI</h1>
  <p style='color:#3d8bcd; font-size:.85rem; letter-spacing:3px'>
    MACHINE LEARNING · LIVE MARKET DATA · NEXT-DAY PREDICTION
  </p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ════════════════════════════════════════════════════════════
# FETCH DATA
# Everything inside try/except so errors render visibly
# ════════════════════════════════════════════════════════════
with st.spinner(f"⚡ Fetching live data for **{ticker}**..."):
    try:
        df_raw = get_stock_data(ticker)
        st.toast(f"✅ Loaded {len(df_raw)} rows for {ticker}", icon="📡")
    except RuntimeError as e:
        st.error(f"❌ **Data fetch failed for `{ticker}`**\n\n{e}")
        st.info(
            "💡 **Checklist:**\n"
            "- Is the ticker symbol valid? (e.g. `AAPL`, `TSLA`, `RELIANCE.NS`)\n"
            "- Wait 60 seconds — yfinance may be rate-limited\n"
            "- Streamlit Cloud sometimes blocks outbound requests"
        )
        st.stop()
    except Exception:
        st.error("❌ Unexpected error during data fetch:")
        st.code(traceback.format_exc())
        st.stop()

# Engineer features — wrapped so any crash shows visibly
try:
    df_eng   = engineer_features(df_raw)
    df_clean = df_eng[FEATURE_COLS + ["Close"]].dropna().copy()
except Exception:
    st.error("❌ Feature engineering failed:")
    st.code(traceback.format_exc())
    st.stop()

if len(df_clean) < 30:
    st.error(
        f"❌ Only **{len(df_clean)}** usable rows after feature engineering "
        f"(need ≥30). The ticker `{ticker}` doesn't have enough history."
    )
    st.stop()

# ════════════════════════════════════════════════════════════
# LIVE METRICS ROW
# ════════════════════════════════════════════════════════════
try:
    latest    = df_raw.iloc[-1]
    prev      = df_raw.iloc[-2]
    price_chg = float(latest["Close"]) - float(prev["Close"])
    pct_chg   = price_chg / float(prev["Close"]) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Close",
              f"${float(latest['Close']):.2f}",
              f"{price_chg:+.2f} ({pct_chg:+.2f}%)")
    c2.metric("📂 Open",  f"${float(latest['Open']):.2f}")
    c3.metric("📊 Range", f"${float(latest['Low']):.2f} – ${float(latest['High']):.2f}")
    c4.metric("📦 Vol",   f"{float(latest['Volume'])/1e6:.2f}M")
    c5.metric("📅 Date",  str(df_raw.index[-1].date()))
except Exception:
    st.warning("Could not render live metrics:")
    st.code(traceback.format_exc())

st.divider()

# ════════════════════════════════════════════════════════════
# PRICE CHART
# ════════════════════════════════════════════════════════════
st.markdown("### 📉 PRICE CHART")

try:
    df_plot = df_eng.dropna(subset=["SMA_20"]).copy()

    fig = make_subplots(
        rows=2 if show_volume else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=.03,
        row_heights=[.75, .25] if show_volume else [1.0],
    )

    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot["Open"],
        high=df_plot["High"],
        low=df_plot["Low"],
        close=df_plot["Close"],
        name="OHLC",
        increasing_line_color="#00ff88",
        decreasing_line_color="#ef4444",
    ), row=1, col=1)

    if show_sma:
        for sma_col, color in [("SMA_20","#f59e0b"), ("SMA_50","#3b82f6"), ("SMA_200","#a855f7")]:
            if sma_col in df_plot.columns:
                fig.add_trace(go.Scatter(
                    x=df_plot.index, y=df_plot[sma_col],
                    name=sma_col, line=dict(color=color, width=1.2)
                ), row=1, col=1)

    if show_bb and "BB_upper" in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["BB_upper"], name="BB Upper",
            line=dict(color="#00d4ff", dash="dash", width=1)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["BB_lower"], name="BB Lower",
            line=dict(color="#00d4ff", dash="dash", width=1),
            fill="tonexty", fillcolor="rgba(0,212,255,.04)"
        ), row=1, col=1)

    if show_volume:
        bar_colors = [
            "#00ff88" if float(c) >= float(o) else "#ef4444"
            for c, o in zip(df_plot["Close"], df_plot["Open"])
        ]
        fig.add_trace(go.Bar(
            x=df_plot.index, y=df_plot["Volume"],
            name="Volume", marker_color=bar_colors, opacity=.5
        ), row=2, col=1)

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

except Exception:
    st.error("❌ Chart rendering failed:")
    st.code(traceback.format_exc())

# ════════════════════════════════════════════════════════════
# PREDICTION SECTION
# ════════════════════════════════════════════════════════════
st.markdown("### 🤖 AI PREDICTION")

if not artifacts["loaded"]:
    st.warning(
        "⚠️ **Model not loaded** — check the Model Info panel in the sidebar "
        "for the exact error and path details."
    )
else:
    pipeline = artifacts["pipeline"]
    scaler   = artifacts["scaler"]

    # Build the (1 × 33) input vector
    try:
        X_live = get_latest_features(df_eng)
    except ValueError as e:
        st.error(f"❌ Feature preparation failed:\n\n{e}")
        st.stop()
    except Exception:
        st.error("❌ Unexpected error building feature vector:")
        st.code(traceback.format_exc())
        st.stop()

    # Shape guard — prevents the "X has N features" crash
    n_expected = scaler.n_features_in_
    n_actual   = X_live.shape[1]
    if n_actual != n_expected:
        st.error(
            f"❌ **Feature count mismatch**\n\n"
            f"- Scaler expects: **{n_expected}** features\n"
            f"- This app built: **{n_actual}** features\n\n"
            "Re-train `stock_prediction_pipeline.py` with the same feature "
            "list as `FEATURE_COLS` in this file."
        )
        st.stop()

    # Scale and predict
    try:
        X_scaled        = scaler.transform(X_live)
        predicted_price = float(pipeline.predict(X_scaled)[0])
    except Exception:
        st.error("❌ Prediction call failed:")
        st.code(traceback.format_exc())
        st.stop()

    # Display prediction result
    current_close = float(df_raw["Close"].iloc[-1])
    delta         = predicted_price - current_close
    pct           = delta / current_close * 100
    direction     = "📈 BULLISH" if delta > 0 else "📉 BEARISH"
    d_color       = "#00ff88" if delta > 0 else "#ef4444"

    st.markdown(f"""
    <div style='border:1px solid {d_color}; border-radius:10px; padding:24px;
                background:rgba(0,0,0,.3); margin:10px 0 20px;
                text-align:center; box-shadow:0 0 20px {d_color}33'>
        <div style='font-family:Orbitron,sans-serif; font-size:14px;
                    color:{d_color}; letter-spacing:4px; margin-bottom:8px;'>
            PREDICTED NEXT-DAY CLOSE
        </div>
        <div style='font-family:Orbitron,sans-serif; font-size:3rem;
                    color:{d_color}; text-shadow:0 0 30px {d_color}'>
            ${predicted_price:.2f}
        </div>
        <div style='color:#888; font-size:13px; margin-top:6px'>
            {direction} &nbsp;|&nbsp;
            <span style='color:{d_color}'>{delta:+.2f} ({pct:+.2f}%)</span>
            &nbsp;vs close of
            <span style='color:#fff'>${current_close:.2f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Historical overlay chart
    try:
        n_hist   = min(120, len(df_clean))
        h_X      = df_clean[FEATURE_COLS].iloc[-n_hist:].values
        h_scaled = scaler.transform(h_X)
        h_pred   = pipeline.predict(h_scaled)
        h_dates  = df_clean.index[-n_hist:]
        h_actual = df_clean["Close"].iloc[-n_hist:].values.flatten()

        next_bday = h_dates[-1] + pd.tseries.offsets.BDay(1)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=h_dates, y=h_actual, name="Actual",
            line=dict(color="#60a5fa", width=2)
        ))
        fig2.add_trace(go.Scatter(
            x=h_dates, y=h_pred, name="Predicted",
            line=dict(color="#f59e0b", width=2, dash="dot")
        ))
        fig2.add_trace(go.Scatter(
            x=[next_bday], y=[predicted_price],
            name="Next-Day Forecast", mode="markers",
            marker=dict(color=d_color, size=14, symbol="diamond",
                        line=dict(color="white", width=1.5))
        ))
        fig2.update_layout(
            title="Actual vs Predicted (Last 120 sessions)",
            template="plotly_dark",
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#0f1628",
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    except Exception:
        st.warning("Could not render prediction overlay chart:")
        st.code(traceback.format_exc())

    # Performance metrics
    st.markdown("### 🏆 MODEL PERFORMANCE")
    meta = artifacts["metadata"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE",      f"${meta.get('test_mae',  'N/A')}")
    m2.metric("RMSE",     f"${meta.get('test_rmse', 'N/A')}")
    m3.metric("R² Score", f"{meta.get('test_r2',   'N/A')}")
    m4.metric("MAPE",     f"{meta.get('test_mape', 'N/A')}%")

    # Debug: show exact values the model received
    with st.expander("🔬 Live Feature Values Sent to Model", expanded=False):
        debug_df = (
            pd.DataFrame(X_live, columns=FEATURE_COLS)
            .T
            .rename(columns={0: "Value"})
        )
        debug_df["Value"] = debug_df["Value"].apply(lambda x: f"{x:.6f}")
        st.dataframe(debug_df, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ════════════════════════════════════════════════════════════
with st.expander("📡 TECHNICAL INDICATORS", expanded=False):
    try:
        ic1, ic2 = st.columns(2)

        with ic1:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=df_eng.index, y=df_eng["RSI_14"],
                name="RSI 14", line=dict(color="#f59e0b", width=1.5)
            ))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef4444")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="#00ff88")
            fig_rsi.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,.05)", line_width=0)
            fig_rsi.add_hrect(y0=0,  y1=30,  fillcolor="rgba(0,255,136,.05)", line_width=0)
            fig_rsi.update_layout(
                title="RSI (14)", template="plotly_dark",
                paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1628",
                height=280, yaxis=dict(range=[0, 100]),
                margin=dict(l=5, r=5, t=40, b=5),
            )
            st.plotly_chart(fig_rsi, use_container_width=True)

        with ic2:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(
                x=df_eng.index, y=df_eng["MACD"],
                name="MACD", line=dict(color="#00d4ff", width=1.5)
            ))
            fig_macd.add_trace(go.Scatter(
                x=df_eng.index, y=df_eng["MACD_signal"],
                name="Signal", line=dict(color="#f59e0b", width=1.2)
            ))
            macd_colors = [
                "#00ff88" if v >= 0 else "#ef4444"
                for v in df_eng["MACD_hist"].fillna(0)
            ]
            fig_macd.add_trace(go.Bar(
                x=df_eng.index, y=df_eng["MACD_hist"],
                name="Histogram", marker_color=macd_colors, opacity=.6
            ))
            fig_macd.update_layout(
                title="MACD", template="plotly_dark",
                paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1628",
                height=280, margin=dict(l=5, r=5, t=40, b=5),
            )
            st.plotly_chart(fig_macd, use_container_width=True)

    except Exception:
        st.warning("Indicator charts failed:")
        st.code(traceback.format_exc())

# ════════════════════════════════════════════════════════════
# RECENT DATA TABLE
# ════════════════════════════════════════════════════════════
with st.expander("📋 RECENT MARKET DATA", expanded=False):
    try:
        disp = df_raw[["Open","High","Low","Close","Volume"]].tail(20).copy()
        disp.index = disp.index.strftime("%Y-%m-%d")
        disp = disp.round(2)
        disp["Volume"] = disp["Volume"].apply(lambda x: f"{float(x)/1e6:.2f}M")
        st.dataframe(disp, use_container_width=True)
    except Exception:
        st.warning("Table render failed:")
        st.code(traceback.format_exc())
