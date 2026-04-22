# ==============================================================
# app.py  —  StockSight AI  (Shape-Verified Edition)
#
# Root cause of "X has 7 features, RobustScaler expects 33":
#   The scaler.transform() was receiving raw OHLCV (7 cols)
#   instead of the fully-engineered 33-column vector.
#
# Fix strategy:
#   1. engineer_features()  ← builds every indicator, returns df
#   2. verify_feature_shape() ← asserts exactly 33 cols, correct order
#   3. get_latest_features() ← extracts (1, 33) array for scaler
#   4. Every stage wrapped in try/except → visible error, not blank screen
# ==============================================================

import streamlit as st                    # ← MUST be imported first

# PAGE CONFIG  — must be the very first st.* call, no exceptions
st.set_page_config(
    page_title="StockSight AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── All remaining imports come AFTER set_page_config ─────────
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    st.error("❌ `yfinance` not installed — add it to requirements.txt"); st.stop()

try:
    import joblib
except ImportError:
    st.error("❌ `joblib` not installed — add it to requirements.txt"); st.stop()

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
except ImportError:
    st.error("❌ `plotly` not installed — add it to requirements.txt"); st.stop()

# ══════════════════════════════════════════════════════════════
# 1.  PATH  RESOLUTION
#     Streamlit Cloud clones to /mount/src/<repo-name>/
#     __file__ → /mount/src/<repo-name>/app.py
#     So  BASE_DIR / "model"  is always correct.
# ══════════════════════════════════════════════════════════════
BASE_DIR  = Path(__file__).resolve().parent   # same folder as app.py
MODEL_DIR = BASE_DIR / "model"                # singular — matches GitHub repo

def mpath(name: str) -> Path:
    return MODEL_DIR / name

# ══════════════════════════════════════════════════════════════
# 2.  CANONICAL 33-COLUMN FEATURE LIST
#
#     Order here must be IDENTICAL to the order used when
#     RobustScaler.fit_transform(X_train) was called in
#     stock_prediction_pipeline.py.
#
#     Group          Columns                                Count
#     ──────────────────────────────────────────────────────────
#     Raw OHLCV      Open High Low Volume                     4
#     SMA            SMA_5 10 20 50                           4
#     EMA            EMA_5 10 20                              3
#     Bollinger      BB_width BB_upper BB_lower               3
#     Oscillators    RSI_14 MACD MACD_signal MACD_hist        4
#     Volatility     Volatility_5 Volatility_20               2
#     Momentum       Momentum_5 Momentum_10                   2
#     Close lags     Close_lag_1 2 3 5                        4
#     Return lags    Return_lag_1 2 3                         3
#     Vol+Calendar   Volume_ratio DayOfWeek Month Quarter     4
#     ──────────────────────────────────────────────────────────
#     TOTAL                                                   33
# ══════════════════════════════════════════════════════════════
FEATURE_COLS: list[str] = [
    # ── Raw price / volume (4) ────────────────────────────────
    "Open", "High", "Low", "Volume",
    # ── Simple Moving Averages (4) ───────────────────────────
    "SMA_5", "SMA_10", "SMA_20", "SMA_50",
    # ── Exponential Moving Averages (3) ──────────────────────
    "EMA_5", "EMA_10", "EMA_20",
    # ── Bollinger Bands (3) ──────────────────────────────────
    "BB_width", "BB_upper", "BB_lower",
    # ── Oscillators (4) ──────────────────────────────────────
    "RSI_14", "MACD", "MACD_signal", "MACD_hist",
    # ── Volatility (2) ───────────────────────────────────────
    "Volatility_5", "Volatility_20",
    # ── Price momentum (2) ───────────────────────────────────
    "Momentum_5", "Momentum_10",
    # ── Close lag features (4) ───────────────────────────────
    "Close_lag_1", "Close_lag_2", "Close_lag_3", "Close_lag_5",
    # ── Return lag features (3) ──────────────────────────────
    "Return_lag_1", "Return_lag_2", "Return_lag_3",
    # ── Volume + calendar (4) ────────────────────────────────
    "Volume_ratio", "DayOfWeek", "Month", "Quarter",
]

# Hard stop at import time — catches any accidental list edits
assert len(FEATURE_COLS) == 33, (
    f"FEATURE_COLS has {len(FEATURE_COLS)} entries — expected exactly 33."
)

# ══════════════════════════════════════════════════════════════
# 3.  CSS  —  Neon-green dark theme
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono
             &family=Orbitron:wght@400;700;900&display=swap');

:root {
    --ng: #00ff88; --nb: #00d4ff;
    --bg: #0a0e1a; --cb: #0f1628; --br: #1e2d4a;
}
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: #c8d8f0 !important;
    font-family: 'Share Tech Mono', monospace !important;
}
section[data-testid="stSidebar"] {
    background: var(--cb) !important;
    border-right: 1px solid var(--br) !important;
}
h1,h2,h3 {
    font-family:'Orbitron',sans-serif !important;
    color:var(--ng) !important;
    text-shadow:0 0 20px rgba(0,255,136,.4) !important;
}
[data-testid="metric-container"] {
    background:var(--cb) !important;
    border:1px solid var(--ng) !important;
    border-radius:8px !important; padding:12px !important;
    box-shadow:0 0 12px rgba(0,255,136,.1) !important;
}
[data-testid="metric-container"] label {
    color:var(--nb) !important; font-size:11px !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color:var(--ng) !important; font-size:22px !important;
    font-family:'Orbitron',sans-serif !important;
}
.stButton>button {
    background:transparent !important;
    border:1px solid var(--ng) !important; color:var(--ng) !important;
    font-family:'Orbitron',sans-serif !important;
    font-size:12px !important; letter-spacing:2px !important;
    transition:all .3s !important; border-radius:4px !important;
}
.stButton>button:hover {
    background:rgba(0,255,136,.1) !important;
    box-shadow:0 0 20px rgba(0,255,136,.3) !important;
}
hr { border-color:var(--br) !important; }
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--ng);border-radius:3px}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 4.  FEATURE  ENGINEERING
#
#     Input  : raw OHLCV DataFrame with DatetimeIndex
#               must have ≥ 215 rows so SMA-200 + lag-10
#               produce at least one non-NaN complete row
#     Output : same df with ALL feature columns appended
#
#     Every indicator group is built in the same order as
#     FEATURE_COLS so there is zero ambiguity about column order.
# ══════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends all 33 technical-indicator columns to df.
    Returns the extended DataFrame (original rows preserved).
    Does NOT drop NaN rows — caller decides when to dropna().
    """
    d = df.copy()

    close  = d["Close"]
    volume = d["Volume"]

    # ── A. Simple Moving Averages ─────────────────────────────
    # windows: 5, 10, 20, 50 go into FEATURE_COLS
    # window 200 is computed but only used for the chart overlay
    for w in [5, 10, 20, 50, 200]:
        d[f"SMA_{w}"] = close.rolling(window=w, min_periods=w).mean()

    # ── B. Exponential Moving Averages ───────────────────────
    for w in [5, 10, 20]:
        d[f"EMA_{w}"] = close.ewm(span=w, adjust=False).mean()

    # ── C. Bollinger Bands (20-day, ±2σ) ─────────────────────
    _bb_mid       = close.rolling(window=20, min_periods=20).mean()
    _bb_std       = close.rolling(window=20, min_periods=20).std()
    d["BB_upper"] = _bb_mid + 2.0 * _bb_std
    d["BB_lower"] = _bb_mid - 2.0 * _bb_std
    # width as fraction of mid (avoids scale dependency)
    d["BB_width"] = (d["BB_upper"] - d["BB_lower"]) / (_bb_mid + 1e-9)

    # ── D. RSI (14-day Wilder smoothing) ─────────────────────
    _delta        = close.diff()
    _gain         = _delta.clip(lower=0).rolling(window=14, min_periods=14).mean()
    _loss         = (-_delta.clip(upper=0)).rolling(window=14, min_periods=14).mean()
    d["RSI_14"]   = 100.0 - (100.0 / (1.0 + _gain / (_loss + 1e-9)))

    # ── E. MACD  (12 / 26 / 9) ───────────────────────────────
    _ema12           = close.ewm(span=12, adjust=False).mean()
    _ema26           = close.ewm(span=26, adjust=False).mean()
    d["MACD"]        = _ema12 - _ema26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"]   = d["MACD"] - d["MACD_signal"]

    # ── F. Daily return + rolling volatility ─────────────────
    _ret               = close.pct_change()
    d["Daily_Return"]  = _ret                                    # helper — not in FEATURE_COLS
    d["Volatility_5"]  = _ret.rolling(window=5,  min_periods=5 ).std()
    d["Volatility_20"] = _ret.rolling(window=20, min_periods=20).std()

    # ── G. Price momentum ────────────────────────────────────
    d["Momentum_5"]  = close - close.shift(5)
    d["Momentum_10"] = close - close.shift(10)

    # ── H. Close-price lag features ──────────────────────────
    for lag in [1, 2, 3, 5]:                   # lag-10 not in FEATURE_COLS
        d[f"Close_lag_{lag}"] = close.shift(lag)

    # ── I. Daily-return lag features ─────────────────────────
    for lag in [1, 2, 3]:
        d[f"Return_lag_{lag}"] = _ret.shift(lag)

    # ── J. Volume ratio (vs 10-day rolling mean) ─────────────
    _vol_ma10         = volume.rolling(window=10, min_periods=10).mean()
    d["Volume_ratio"] = volume / (_vol_ma10 + 1e-9)

    # ── K. Calendar features (no NaN, always available) ──────
    d["DayOfWeek"] = d.index.dayofweek     # 0=Mon … 4=Fri
    d["Month"]     = d.index.month
    d["Quarter"]   = d.index.quarter

    return d


# ══════════════════════════════════════════════════════════════
# 5.  SHAPE  VERIFICATION
#     Call this right before scaler.transform() to guarantee
#     the exact (1, 33) array with columns in FEATURE_COLS order.
# ══════════════════════════════════════════════════════════════
def verify_feature_shape(df_engineered: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Parameters
    ----------
    df_engineered : output of engineer_features()

    Returns
    -------
    X_latest : np.ndarray of shape (1, 33) — latest complete row
    clean_df : rows-only-no-NaN slice used for the overlay chart

    Raises
    ------
    ValueError with a diagnostic table if anything is wrong
    """
    # ── Step 1: check all 33 columns exist ───────────────────
    missing = [c for c in FEATURE_COLS if c not in df_engineered.columns]
    if missing:
        raise ValueError(
            f"engineer_features() did not produce {len(missing)} required column(s):\n"
            + "\n".join(f"  • {c}" for c in missing)
        )

    # ── Step 2: select in canonical order ────────────────────
    feature_df = df_engineered[FEATURE_COLS]          # shape (N, 33)

    # ── Step 3: drop NaN rows (rolling-window warmup) ────────
    clean_df = feature_df.dropna()

    if clean_df.empty:
        raise ValueError(
            "Every row has at least one NaN after feature engineering.\n"
            f"  DataFrame has {len(df_engineered)} total rows — need ≥ 215.\n"
            "  Try period='2y' or use a well-traded ticker like AAPL."
        )

    # ── Step 4: extract the single most-recent complete row ──
    X_latest = clean_df.iloc[[-1]].values            # shape (1, 33)

    # ── Step 5: hard assert — will surface any silent bugs ───
    if X_latest.shape != (1, 33):
        raise ValueError(
            f"Shape after extraction is {X_latest.shape}, expected (1, 33).\n"
            "This should never happen — please report this as a bug."
        )

    if np.any(np.isnan(X_latest)):
        raise ValueError(
            "The latest row still contains NaN values even after dropna().\n"
            "Possible cause: a feature column was all-NaN (e.g. insufficient data)."
        )

    # also return the Close column aligned with clean rows for the overlay chart
    close_clean = df_engineered["Close"].reindex(clean_df.index)
    clean_df    = clean_df.copy()
    clean_df["Close"] = close_clean

    return X_latest, clean_df


# ══════════════════════════════════════════════════════════════
# 6.  MODEL  LOADING
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_artifacts() -> dict:
    result: dict = {"loaded": False, "error": None}

    pipeline_f = mpath("stock_pipeline_compressed.sav")
    scaler_f   = mpath("scaler.sav")
    meta_f     = mpath("pipeline_metadata.sav")

    # Report every missing file explicitly
    missing_files = [str(f) for f in [pipeline_f, scaler_f] if not f.exists()]
    if missing_files:
        result["error"] = (
            "**Model file(s) not found:**\n"
            + "\n".join(f"  • `{p}`" for p in missing_files)
            + "\n\n**Fix checklist:**\n"
            "1. Folder must be named `model` (singular, lowercase)\n"
            "2. `.sav` files must be committed — not listed in `.gitignore`\n"
            "3. Re-run `stock_prediction_pipeline.py`, then `git push`"
        )
        return result

    try:
        result["pipeline"] = joblib.load(pipeline_f)
        result["scaler"]   = joblib.load(scaler_f)
        result["metadata"] = joblib.load(meta_f) if meta_f.exists() else {}
        result["loaded"]   = True
    except Exception:
        result["error"] = (
            "**Exception loading model files:**\n\n"
            f"```\n{traceback.format_exc()}\n```"
        )
        return result

    # Cross-check: scaler must expect 33 features
    n = result["scaler"].n_features_in_
    if n != 33:
        result["loaded"] = False
        result["error"]  = (
            f"**Scaler mismatch:** loaded scaler expects **{n}** features "
            f"but this app always sends **33**.\n\n"
            "Re-train `stock_prediction_pipeline.py` using the same "
            "FEATURE_COLS list as this file, then redeploy."
        )

    return result


# ══════════════════════════════════════════════════════════════
# 7.  DATA  FETCHING
#     period='2y' → ~504 trading days, well above the 215 needed
# ══════════════════════════════════════════════════════════════
MIN_ROWS = 215   # 200 (SMA-200) + 15 buffer for lag/volatility warmup

@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(ticker: str) -> pd.DataFrame:
    periods  = ["2y", "3y", "5y"]
    last_err: Exception | None = None

    for attempt, period in enumerate(periods, start=1):
        try:
            raw = yf.download(
                ticker,
                period=period,
                interval="1d",
                progress=False,
                auto_adjust=True,
                timeout=20,
            )

            # yfinance >= 0.2.x sometimes returns multi-level columns
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            if raw.empty:
                raise ValueError(f"yfinance returned an empty DataFrame for '{ticker}'")

            raw.index = pd.to_datetime(raw.index)
            raw.sort_index(inplace=True)
            raw.ffill(inplace=True)
            raw.bfill(inplace=True)

            if len(raw) < MIN_ROWS:
                raise ValueError(
                    f"Only {len(raw)} rows returned — need ≥ {MIN_ROWS}. "
                    f"Trying longer period ({period} → next)…"
                )

            return raw

        except Exception as exc:
            last_err = exc
            if attempt < len(periods):
                time.sleep(2 ** attempt)   # 2 s, 4 s back-off

    raise RuntimeError(
        f"Could not fetch data for '{ticker}' after {len(periods)} attempts.\n"
        f"Last error: {last_err}\n\n"
        "Possible causes: invalid ticker · API rate-limit · network restriction"
    )


# ══════════════════════════════════════════════════════════════
# 8.  SIDEBAR
# ══════════════════════════════════════════════════════════════
artifacts = load_artifacts()   # load once, referenced everywhere below

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
    st.button("🚀 RUN PREDICTION", use_container_width=True)

    st.divider()
    st.markdown("### 📋 Model Info")

    if artifacts["loaded"]:
        meta = artifacts["metadata"]
        st.success("✅ Pipeline Loaded")
        st.markdown(f"`Folder :` `{MODEL_DIR.name}/`")
        st.markdown(f"`Features:` **{artifacts['scaler'].n_features_in_}** ✓")
        st.markdown(f"`Ticker  :` **{meta.get('ticker', 'N/A')}**")
        st.markdown(f"`Cutoff  :` **{meta.get('train_end_date', 'N/A')}**")
        st.markdown(f"`R²      :` **{meta.get('test_r2', 'N/A')}**")
        st.markdown(f"`MAE     :` **${meta.get('test_mae', 'N/A')}**")
    else:
        st.error("❌ Pipeline not loaded")
        st.markdown(artifacts.get("error", "Unknown error."))

    st.divider()
    st.markdown(
        "<small style='color:#3d5a80'>yfinance · sklearn · Streamlit</small>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════
# 9.  MAIN  PAGE  HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center;padding:20px 0 10px'>
  <h1 style='font-size:2.8rem;letter-spacing:6px'>📈 STOCKSIGHT AI</h1>
  <p style='color:#3d8bcd;font-size:.85rem;letter-spacing:3px'>
    MACHINE LEARNING · LIVE MARKET DATA · NEXT-DAY PREDICTION
  </p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ══════════════════════════════════════════════════════════════
# 10.  FETCH  →  ENGINEER  →  VERIFY
#      Each stage in its own try/except so we always render
#      something useful instead of a blank screen.
# ══════════════════════════════════════════════════════════════

# ── Stage A : fetch raw OHLCV ─────────────────────────────────
with st.spinner(f"⚡ Fetching data for **{ticker}**…"):
    try:
        df_raw = get_stock_data(ticker)
    except RuntimeError as exc:
        st.error(f"❌ **Data fetch failed for `{ticker}`**\n\n{exc}")
        st.info(
            "💡 **Checklist:**\n"
            "- Valid ticker? (`AAPL`, `TSLA`, `INFY.NS` …)\n"
            "- Wait 60 s — yfinance may be rate-limited\n"
            "- Streamlit Cloud can block outbound network calls"
        )
        st.stop()
    except Exception:
        st.error("❌ Unexpected data-fetch error:")
        st.code(traceback.format_exc())
        st.stop()

# ── Stage B : engineer all 33 features ───────────────────────
try:
    df_eng = engineer_features(df_raw)
except Exception:
    st.error("❌ Feature engineering crashed:")
    st.code(traceback.format_exc())
    st.stop()

# ── Stage C : verify shape, extract (1, 33) array ────────────
try:
    X_live, df_clean = verify_feature_shape(df_eng)
except ValueError as exc:
    st.error(f"❌ **Feature shape verification failed:**\n\n{exc}")

    # Show a diagnostic table so the user can see which cols
    # exist and which are all-NaN
    with st.expander("🔬 Feature Diagnostic", expanded=True):
        diag_rows = []
        for col in FEATURE_COLS:
            if col in df_eng.columns:
                n_valid = int(df_eng[col].notna().sum())
                last_v  = df_eng[col].dropna().iloc[-1] if n_valid else float("nan")
                diag_rows.append({
                    "Feature":    col,
                    "Non-NaN rows": n_valid,
                    "Latest value": f"{last_v:.4f}" if not np.isnan(last_v) else "NaN ⚠️",
                    "Status": "✅" if n_valid >= 1 else "❌ ALL NaN",
                })
            else:
                diag_rows.append({
                    "Feature":    col,
                    "Non-NaN rows": 0,
                    "Latest value": "MISSING",
                    "Status": "❌ COLUMN ABSENT",
                })
        st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)
    st.stop()
except Exception:
    st.error("❌ Unexpected error in verify_feature_shape():")
    st.code(traceback.format_exc())
    st.stop()

# ── All clear: show a subtle confirmation ────────────────────
st.success(
    f"✅ `{ticker}` — "
    f"{len(df_raw)} raw rows fetched · "
    f"{len(df_clean)} complete rows after feature engineering · "
    f"input shape = **{X_live.shape}**"
)

# ══════════════════════════════════════════════════════════════
# 11.  LIVE  METRICS  ROW
# ══════════════════════════════════════════════════════════════
try:
    latest    = df_raw.iloc[-1]
    prev      = df_raw.iloc[-2]
    chg       = float(latest["Close"]) - float(prev["Close"])
    pct_chg   = chg / float(prev["Close"]) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Close",
              f"${float(latest['Close']):.2f}",
              f"{chg:+.2f} ({pct_chg:+.2f}%)")
    c2.metric("📂 Open",   f"${float(latest['Open']):.2f}")
    c3.metric("📊 Range",  f"${float(latest['Low']):.2f} – ${float(latest['High']):.2f}")
    c4.metric("📦 Volume", f"{float(latest['Volume'])/1e6:.2f}M")
    c5.metric("📅 Date",   str(df_raw.index[-1].date()))
except Exception:
    st.warning("Metrics row failed:")
    st.code(traceback.format_exc())

st.divider()

# ══════════════════════════════════════════════════════════════
# 12.  PRICE  CHART
# ══════════════════════════════════════════════════════════════
st.markdown("### 📉 PRICE CHART")

try:
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
        for col, color in [
            ("SMA_20", "#f59e0b"),
            ("SMA_50", "#3b82f6"),
            ("SMA_200","#a855f7"),
        ]:
            if col in df_plot.columns:
                fig.add_trace(go.Scatter(
                    x=df_plot.index, y=df_plot[col],
                    name=col, line=dict(color=color, width=1.2)
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
        template="plotly_dark", paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1628",
        height=550, xaxis_rangeslider_visible=False,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

except Exception:
    st.error("❌ Price chart failed:")
    st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════
# 13.  PREDICTION
# ══════════════════════════════════════════════════════════════
st.markdown("### 🤖 AI PREDICTION")

if not artifacts["loaded"]:
    st.warning(
        "⚠️ **Model not loaded** — see the sidebar for the exact error and path details."
    )
else:
    pipeline = artifacts["pipeline"]
    scaler   = artifacts["scaler"]

    # ── 13a.  Final shape guard before scaler.transform() ────
    n_expected = scaler.n_features_in_
    n_actual   = X_live.shape[1]

    if n_actual != n_expected:
        # This should NEVER fire because verify_feature_shape() already
        # guaranteed (1, 33) and load_artifacts() confirmed the scaler
        # expects 33.  If it does fire, show the clearest possible message.
        st.error(
            f"❌ **Shape mismatch — `scaler.transform()` blocked**\n\n"
            f"| | Count |\n|---|---|\n"
            f"| Input features built | **{n_actual}** |\n"
            f"| Scaler expects | **{n_expected}** |\n\n"
            "**How to fix:**\n"
            "1. Open `stock_prediction_pipeline.py`\n"
            "2. Print the exact column order passed to `scaler.fit_transform()`\n"
            "3. Copy that list into `FEATURE_COLS` in this file\n"
            "4. Redeploy"
        )
        with st.expander("🔬 Current FEATURE_COLS (what app.py sends)", expanded=True):
            st.code("\n".join(f"{i+1:>2}. {c}" for i, c in enumerate(FEATURE_COLS)))
        st.stop()

    # ── 13b.  Scale → predict ─────────────────────────────────
    try:
        X_scaled        = scaler.transform(X_live)        # (1, 33) → scaled
        predicted_price = float(pipeline.predict(X_scaled)[0])
    except Exception:
        st.error("❌ `pipeline.predict()` raised an exception:")
        st.code(traceback.format_exc())
        st.stop()

    # ── 13c.  Display prediction card ────────────────────────
    current_close = float(df_raw["Close"].iloc[-1])
    delta         = predicted_price - current_close
    pct           = delta / current_close * 100
    bullish       = delta > 0
    direction     = "📈 BULLISH" if bullish else "📉 BEARISH"
    d_color       = "#00ff88"   if bullish else "#ef4444"

    st.markdown(f"""
    <div style='border:1px solid {d_color};border-radius:10px;padding:28px;
                background:rgba(0,0,0,.35);margin:10px 0 24px;text-align:center;
                box-shadow:0 0 24px {d_color}33'>
        <div style='font-family:Orbitron,sans-serif;font-size:13px;
                    color:{d_color};letter-spacing:4px;margin-bottom:10px'>
            PREDICTED NEXT-DAY CLOSE
        </div>
        <div style='font-family:Orbitron,sans-serif;font-size:3.2rem;
                    color:{d_color};text-shadow:0 0 30px {d_color}'>
            ${predicted_price:.2f}
        </div>
        <div style='color:#888;font-size:13px;margin-top:8px'>
            {direction}&nbsp;|&nbsp;
            <span style='color:{d_color}'>{delta:+.2f}&nbsp;({pct:+.2f}%)</span>
            &nbsp;vs&nbsp;current&nbsp;close&nbsp;
            <span style='color:#fff'>${current_close:.2f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 13d.  Actual vs Predicted overlay ────────────────────
    try:
        n_hist   = min(120, len(df_clean))
        h_X      = df_clean[FEATURE_COLS].iloc[-n_hist:].values   # (n, 33)
        h_scaled = scaler.transform(h_X)
        h_pred   = pipeline.predict(h_scaled)
        h_dates  = df_clean.index[-n_hist:]
        h_actual = df_clean["Close"].iloc[-n_hist:].values.flatten()
        next_bd  = h_dates[-1] + pd.tseries.offsets.BDay(1)

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
            x=[next_bd], y=[predicted_price],
            name="Next-Day Forecast", mode="markers",
            marker=dict(color=d_color, size=14, symbol="diamond",
                        line=dict(color="white", width=1.5))
        ))
        fig2.update_layout(
            title="Actual vs Predicted — last 120 sessions",
            template="plotly_dark", paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1628",
            height=400, margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)
    except Exception:
        st.warning("Overlay chart failed:")
        st.code(traceback.format_exc())

    # ── 13e.  Performance metrics ─────────────────────────────
    st.markdown("### 🏆 MODEL PERFORMANCE (Test Set)")
    meta = artifacts["metadata"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE",      f"${meta.get('test_mae',  'N/A')}")
    m2.metric("RMSE",     f"${meta.get('test_rmse', 'N/A')}")
    m3.metric("R² Score", f"{meta.get('test_r2',   'N/A')}")
    m4.metric("MAPE",     f"{meta.get('test_mape', 'N/A')}%")

    # ── 13f.  Feature debug table ─────────────────────────────
    with st.expander("🔬 Live Feature Values Sent to Scaler", expanded=False):
        dbg = pd.DataFrame(X_live, columns=FEATURE_COLS).T.rename(columns={0: "Live Value"})
        dbg["Live Value"] = dbg["Live Value"].map(lambda x: f"{x:.6f}")
        st.dataframe(dbg, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# 14.  TECHNICAL  INDICATORS  EXPANDER
# ══════════════════════════════════════════════════════════════
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
            macd_colors = [
                "#00ff88" if v >= 0 else "#ef4444"
                for v in df_eng["MACD_hist"].fillna(0)
            ]
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(
                x=df_eng.index, y=df_eng["MACD"],
                name="MACD", line=dict(color="#00d4ff", width=1.5)
            ))
            fig_macd.add_trace(go.Scatter(
                x=df_eng.index, y=df_eng["MACD_signal"],
                name="Signal", line=dict(color="#f59e0b", width=1.2)
            ))
            fig_macd.add_trace(go.Bar(
                x=df_eng.index, y=df_eng["MACD_hist"],
                name="Histogram", marker_color=macd_colors, opacity=.6
            ))
            fig_macd.update_layout(
                title="MACD (12/26/9)", template="plotly_dark",
                paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1628",
                height=280, margin=dict(l=5, r=5, t=40, b=5),
            )
            st.plotly_chart(fig_macd, use_container_width=True)

    except Exception:
        st.warning("Indicator charts failed:")
        st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════
# 15.  RECENT  DATA  TABLE
# ══════════════════════════════════════════════════════════════
with st.expander("📋 RECENT MARKET DATA", expanded=False):
    try:
        disp = df_raw[["Open","High","Low","Close","Volume"]].tail(20).copy()
        disp.index = disp.index.strftime("%Y-%m-%d")
        disp = disp.round(2)
        disp["Volume"] = disp["Volume"].apply(lambda x: f"{float(x)/1e6:.2f}M")
        st.dataframe(disp, use_container_width=True)
    except Exception:
        st.warning("Table failed:")
        st.code(traceback.format_exc())
