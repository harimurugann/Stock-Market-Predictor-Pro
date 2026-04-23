# ==============================================================
# app.py  —  StockSight AI  (Live Tracking Edition)
#
# New in this version
# ───────────────────
#  • Sidebar  : selectbox ticker list + history slider + live-tracking toggle
#  • Metrics  : Current Price, Change ($), Change (%) before the chart
#  • Auto-refresh : time.sleep(60) + st.rerun() when live tracking is ON
#
# Unchanged (model-critical)
# ──────────────────────────
#  • engineer_features()   → produces exactly 33 columns
#  • FEATURE_COLS list     → same order as scaler.fit_transform()
#  • verify_feature_shape()→ asserts (1, 33), no NaN, before scaler
# ==============================================================

import streamlit as st                        # ← must be first import

# PAGE CONFIG — must be the very first st.* call, no exceptions
st.set_page_config(
    page_title="StockSight AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── All other imports come AFTER set_page_config ─────────────
import warnings
warnings.filterwarnings("ignore")

import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    st.error("❌ `yfinance` not installed — add it to requirements.txt")
    st.stop()

try:
    import joblib
except ImportError:
    st.error("❌ `joblib` not installed — add it to requirements.txt")
    st.stop()

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
except ImportError:
    st.error("❌ `plotly` not installed — add it to requirements.txt")
    st.stop()

# ══════════════════════════════════════════════════════════════
# PATH RESOLUTION
# Streamlit Cloud: /mount/src/<repo-name>/app.py
# LOCAL          : wherever app.py lives
# MODEL_DIR      : always  <app.py folder> / "model"  (singular)
# ══════════════════════════════════════════════════════════════
BASE_DIR  = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

def mpath(name: str) -> Path:
    return MODEL_DIR / name

# ══════════════════════════════════════════════════════════════
# CANONICAL 33-COLUMN FEATURE LIST
# Order here == order passed to RobustScaler.fit_transform()
# in stock_prediction_pipeline.py — do NOT reorder.
#
#  Group            Columns                              Count
#  ─────────────────────────────────────────────────────────
#  Raw OHLCV        Open High Low Volume                   4
#  SMA              SMA_5 10 20 50                         4
#  EMA              EMA_5 10 20                            3
#  Bollinger        BB_width BB_upper BB_lower             3
#  Oscillators      RSI_14 MACD MACD_signal MACD_hist      4
#  Volatility       Volatility_5 Volatility_20             2
#  Momentum         Momentum_5 Momentum_10                 2
#  Close lags       Close_lag_1 2 3 5                      4
#  Return lags      Return_lag_1 2 3                       3
#  Vol + Calendar   Volume_ratio DayOfWeek Month Quarter   4
#  ─────────────────────────────────────────────────────────
#  TOTAL                                                   33
# ══════════════════════════════════════════════════════════════
FEATURE_COLS: list[str] = [
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

assert len(FEATURE_COLS) == 33, (
    f"FEATURE_COLS has {len(FEATURE_COLS)} entries — must be exactly 33."
)

# ══════════════════════════════════════════════════════════════
# TICKER CATALOGUE
# Add / remove tickers here freely — the rest of the app adapts.
# ══════════════════════════════════════════════════════════════
TICKER_OPTIONS: list[str] = [
    # US mega-cap
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    # Finance / diversified
    "JPM", "BRK-B", "V", "JNJ", "XOM",
    # Indian markets (NSE)
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS",
]

# ══════════════════════════════════════════════════════════════
# CSS — neon-green dark theme
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono
             &family=Orbitron:wght@400;700;900&display=swap');
:root{--ng:#00ff88;--nb:#00d4ff;--bg:#0a0e1a;--cb:#0f1628;--br:#1e2d4a}
html,body,[class*="css"]{
    background-color:var(--bg)!important;
    color:#c8d8f0!important;
    font-family:'Share Tech Mono',monospace!important}
section[data-testid="stSidebar"]{
    background:var(--cb)!important;
    border-right:1px solid var(--br)!important}
h1,h2,h3{
    font-family:'Orbitron',sans-serif!important;
    color:var(--ng)!important;
    text-shadow:0 0 20px rgba(0,255,136,.4)!important}
[data-testid="metric-container"]{
    background:var(--cb)!important;
    border:1px solid var(--ng)!important;
    border-radius:8px!important;padding:12px!important;
    box-shadow:0 0 12px rgba(0,255,136,.1)!important}
[data-testid="metric-container"] label{
    color:var(--nb)!important;font-size:11px!important}
[data-testid="metric-container"] [data-testid="metric-value"]{
    color:var(--ng)!important;font-size:22px!important;
    font-family:'Orbitron',sans-serif!important}
.stButton>button{
    background:transparent!important;
    border:1px solid var(--ng)!important;color:var(--ng)!important;
    font-family:'Orbitron',sans-serif!important;
    font-size:12px!important;letter-spacing:2px!important;
    transition:all .3s!important;border-radius:4px!important}
.stButton>button:hover{
    background:rgba(0,255,136,.1)!important;
    box-shadow:0 0 20px rgba(0,255,136,.3)!important}
/* live-tracking badge */
.live-badge{
    display:inline-block;padding:3px 10px;border-radius:20px;
    font-size:11px;letter-spacing:2px;font-family:'Orbitron',sans-serif;
    background:rgba(0,255,136,.12);border:1px solid #00ff88;color:#00ff88;
    animation:pulse 1.5s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.45}}
hr{border-color:var(--br)!important}
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--ng);border-radius:3px}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SECTION 1 — SIDEBAR CONFIGURATION
# Order: ticker selectbox → history slider → chart toggles
#         → run-prediction button → live-tracking toggle
#         → refresh-interval slider → model info
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ STOCKSIGHT AI")
    st.divider()

    # ── 1a. Ticker selection ──────────────────────────────────
    st.markdown("### 🔧 Configuration")

    ticker = st.selectbox(
        "Select Ticker",
        options=TICKER_OPTIONS,
        index=0,                       # default: AAPL
        help="Choose a stock. Indian NSE tickers end in .NS",
    )

    # Allow typing a custom ticker not in the list
    custom = st.text_input(
        "Or type a custom ticker",
        value="",
        max_chars=15,
        placeholder="e.g. AMZN, NFLX, SBIN.NS",
    ).upper().strip()

    if custom:
        ticker = custom                # custom input overrides selectbox

    st.caption(f"Active ticker: **{ticker}**")

    # ── 1b. Data history slider ───────────────────────────────
    st.divider()
    st.markdown("### 📅 History Period")

    history_days = st.slider(
        "Days of history to fetch",
        min_value=250,        # 250 = absolute minimum for SMA-200 + lags
        max_value=1825,       # 5 years
        value=730,            # default: 2 years
        step=50,
        help=(
            "Minimum 250 days required so SMA-200 and all lag features "
            "have enough data. More history = better indicator quality."
        ),
    )
    st.caption(f"Fetching ~{history_days} calendar days (~{int(history_days*5/7)} trading days)")

    # ── 1c. Chart display toggles ─────────────────────────────
    st.divider()
    st.markdown("### 📊 Chart Settings")
    show_sma    = st.checkbox("Show Moving Averages", value=True)
    show_bb     = st.checkbox("Show Bollinger Bands",  value=True)
    show_volume = st.checkbox("Show Volume",            value=True)

    # ── 1d. Run-prediction button ─────────────────────────────
    st.divider()
    st.button("🚀 RUN PREDICTION", use_container_width=True)

    # ── 1e. Live tracking toggle + refresh interval ───────────
    st.divider()
    st.markdown("### 🔴 Live Refresh")

    live_tracking = st.toggle(
        "Enable Live Tracking",
        value=False,
        help="Auto-refreshes the app every N seconds to show the latest price.",
    )

    refresh_secs = st.slider(
        "Refresh every (seconds)",
        min_value=30,
        max_value=300,
        value=60,
        step=10,
        disabled=not live_tracking,
        help="How often to re-fetch data. Values < 30 s may trigger yfinance rate limits.",
    )

    if live_tracking:
        st.markdown(
            "<div class='live-badge'>● LIVE</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Refreshing every {refresh_secs} s")

    # ── 1f. Model info ────────────────────────────────────────
    st.divider()
    st.markdown("### 📋 Model Info")

    # (artifacts loaded below — placeholder shown now, filled after load)
    model_info_slot = st.empty()

    st.divider()
    st.markdown(
        "<small style='color:#3d5a80'>yfinance · sklearn · Streamlit</small>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING  (unchanged — model depends on this)
# ══════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends all 33 technical-indicator columns to df.
    Returns extended DataFrame; does NOT drop NaN rows.
    Requires ≥ 215 rows so every rolling window has enough data.
    """
    d = df.copy()
    close  = d["Close"]
    volume = d["Volume"]

    # A. Simple Moving Averages (200 kept for chart overlay)
    for w in [5, 10, 20, 50, 200]:
        d[f"SMA_{w}"] = close.rolling(window=w, min_periods=w).mean()

    # B. Exponential Moving Averages
    for w in [5, 10, 20]:
        d[f"EMA_{w}"] = close.ewm(span=w, adjust=False).mean()

    # C. Bollinger Bands  (20-day ±2σ)
    _bb_mid       = close.rolling(window=20, min_periods=20).mean()
    _bb_std       = close.rolling(window=20, min_periods=20).std()
    d["BB_upper"] = _bb_mid + 2.0 * _bb_std
    d["BB_lower"] = _bb_mid - 2.0 * _bb_std
    d["BB_width"] = (d["BB_upper"] - d["BB_lower"]) / (_bb_mid + 1e-9)

    # D. RSI  (14-day)
    _delta      = close.diff()
    _gain       = _delta.clip(lower=0).rolling(window=14, min_periods=14).mean()
    _loss       = (-_delta.clip(upper=0)).rolling(window=14, min_periods=14).mean()
    d["RSI_14"] = 100.0 - (100.0 / (1.0 + _gain / (_loss + 1e-9)))

    # E. MACD  (12 / 26 / 9)
    _ema12           = close.ewm(span=12, adjust=False).mean()
    _ema26           = close.ewm(span=26, adjust=False).mean()
    d["MACD"]        = _ema12 - _ema26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"]   = d["MACD"] - d["MACD_signal"]

    # F. Returns + Volatility
    _ret               = close.pct_change()
    d["Daily_Return"]  = _ret
    d["Volatility_5"]  = _ret.rolling(window=5,  min_periods=5 ).std()
    d["Volatility_20"] = _ret.rolling(window=20, min_periods=20).std()

    # G. Price Momentum
    d["Momentum_5"]  = close - close.shift(5)
    d["Momentum_10"] = close - close.shift(10)

    # H. Close-price lags
    for lag in [1, 2, 3, 5]:
        d[f"Close_lag_{lag}"] = close.shift(lag)

    # I. Daily-return lags
    for lag in [1, 2, 3]:
        d[f"Return_lag_{lag}"] = _ret.shift(lag)

    # J. Volume ratio
    _vol_ma10         = volume.rolling(window=10, min_periods=10).mean()
    d["Volume_ratio"] = volume / (_vol_ma10 + 1e-9)

    # K. Calendar
    d["DayOfWeek"] = d.index.dayofweek
    d["Month"]     = d.index.month
    d["Quarter"]   = d.index.quarter

    return d


# ══════════════════════════════════════════════════════════════
# SHAPE VERIFICATION  (unchanged — model depends on this)
# ══════════════════════════════════════════════════════════════
def verify_feature_shape(
    df_engineered: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Returns (X_latest, clean_df) where X_latest.shape == (1, 33).
    Raises ValueError with a diagnostic message if anything is wrong.
    """
    missing = [c for c in FEATURE_COLS if c not in df_engineered.columns]
    if missing:
        raise ValueError(
            f"engineer_features() did not produce {len(missing)} column(s):\n"
            + "\n".join(f"  • {c}" for c in missing)
        )

    feature_df = df_engineered[FEATURE_COLS]
    clean_df   = feature_df.dropna()

    if clean_df.empty:
        raise ValueError(
            "Zero complete rows after dropping NaN.\n"
            f"DataFrame has {len(df_engineered)} rows — need ≥ 215.\n"
            "Increase history_days or choose a more liquid ticker."
        )

    X_latest = clean_df.iloc[[-1]].values          # (1, 33)

    if X_latest.shape != (1, 33):
        raise ValueError(
            f"Shape after extraction: {X_latest.shape} — expected (1, 33)."
        )
    if np.any(np.isnan(X_latest)):
        raise ValueError(
            "Latest feature row still contains NaN after dropna().\n"
            "One or more indicators is entirely NaN (insufficient data)."
        )

    close_clean          = df_engineered["Close"].reindex(clean_df.index)
    clean_df             = clean_df.copy()
    clean_df["Close"]    = close_clean
    return X_latest, clean_df


# ══════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_artifacts() -> dict:
    result: dict = {"loaded": False, "error": None}

    pipeline_f = mpath("stock_pipeline_compressed.sav")
    scaler_f   = mpath("scaler.sav")
    meta_f     = mpath("pipeline_metadata.sav")

    missing = [str(f) for f in [pipeline_f, scaler_f] if not f.exists()]
    if missing:
        result["error"] = (
            "**Model file(s) not found:**\n"
            + "\n".join(f"  • `{p}`" for p in missing)
            + "\n\n**Fix checklist:**\n"
            "1. Folder must be `model/` (singular)\n"
            "2. `.sav` files must be committed (not in `.gitignore`)\n"
            "3. Re-run `stock_prediction_pipeline.py` then `git push`"
        )
        return result

    try:
        result["pipeline"] = joblib.load(pipeline_f)
        result["scaler"]   = joblib.load(scaler_f)
        result["metadata"] = joblib.load(meta_f) if meta_f.exists() else {}
        result["loaded"]   = True
    except Exception:
        result["error"] = f"```\n{traceback.format_exc()}\n```"
        return result

    n = result["scaler"].n_features_in_
    if n != 33:
        result["loaded"] = False
        result["error"]  = (
            f"Scaler expects **{n}** features but app sends **33**.\n"
            "Re-train `stock_prediction_pipeline.py` to sync feature lists."
        )
    return result


artifacts = load_artifacts()

# Fill the sidebar model-info slot now that artifacts are loaded
with model_info_slot:
    if artifacts["loaded"]:
        meta = artifacts["metadata"]
        st.success("✅ Pipeline Loaded")
        st.markdown(f"`Folder  :` `{MODEL_DIR.name}/`")
        st.markdown(f"`Features:` **{artifacts['scaler'].n_features_in_}** ✓")
        st.markdown(f"`Ticker  :` **{meta.get('ticker','N/A')}**")
        st.markdown(f"`Cutoff  :` **{meta.get('train_end_date','N/A')}**")
        st.markdown(f"`R²      :` **{meta.get('test_r2','N/A')}**")
        st.markdown(f"`MAE     :` **${meta.get('test_mae','N/A')}**")
    else:
        st.error("❌ Pipeline not loaded")
        st.markdown(artifacts.get("error", ""))


# ══════════════════════════════════════════════════════════════
# SECTION 2 — DYNAMIC DATA FETCHING
# Uses `ticker` and `history_days` from the sidebar widgets.
# MIN_ROWS guard ensures enough warmup for SMA-200 + all lags.
# ══════════════════════════════════════════════════════════════
MIN_ROWS = 215    # 200 (SMA-200) + 15 buffer

@st.cache_data(ttl=60, show_spinner=False)   # 60-s cache aligns with live refresh
def get_stock_data(ticker: str, history_days: int) -> pd.DataFrame:
    """
    Fetch `history_days` calendar days of daily OHLCV for `ticker`.
    Retries with escalating periods if the first attempt returns too few rows.
    """
    # Convert calendar days → approximate trading days for yfinance period strings
    trading_days = int(history_days * 5 / 7)

    # Build a list of period strings, ensuring the first covers history_days
    def days_to_period(d: int) -> str:
        if   d <= 365:  return "1y"
        elif d <= 730:  return "2y"
        elif d <= 1095: return "3y"
        else:           return "5y"

    periods   = [days_to_period(history_days), "3y", "5y"]
    # deduplicate while preserving order
    seen, unique_periods = set(), []
    for p in periods:
        if p not in seen:
            seen.add(p); unique_periods.append(p)

    last_err: Exception | None = None

    for attempt, period in enumerate(unique_periods, start=1):
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
                    f"Only {len(raw)} rows (need ≥{MIN_ROWS}). "
                    f"Trying longer period…"
                )
            return raw

        except Exception as exc:
            last_err = exc
            if attempt < len(unique_periods):
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Could not fetch data for '{ticker}' after {len(unique_periods)} attempts.\n"
        f"Last error: {last_err}"
    )


# ══════════════════════════════════════════════════════════════
# MAIN PAGE HEADER
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='text-align:center;padding:20px 0 8px'>
  <h1 style='font-size:2.8rem;letter-spacing:6px'>📈 STOCKSIGHT AI</h1>
  <p style='color:#3d8bcd;font-size:.85rem;letter-spacing:3px'>
    MACHINE LEARNING · LIVE MARKET DATA · NEXT-DAY PREDICTION
  </p>
  {'<div class="live-badge" style="margin:8px auto;width:fit-content">● LIVE TRACKING ON</div>' if live_tracking else ''}
</div>
""", unsafe_allow_html=True)
st.divider()

# ── Fetch data ───────────────────────────────────────────────
with st.spinner(f"⚡ Fetching {history_days}-day history for **{ticker}**…"):
    try:
        df_raw = get_stock_data(ticker, history_days)
    except RuntimeError as exc:
        st.error(f"❌ **Data fetch failed for `{ticker}`**\n\n{exc}")
        st.info(
            "💡 **Checklist:**\n"
            "- Valid ticker? (`AAPL`, `TSLA`, `INFY.NS` …)\n"
            "- Wait 60 s — yfinance may be rate-limited\n"
            "- Increase history slider above 250 days"
        )
        st.stop()
    except Exception:
        st.error("❌ Unexpected fetch error:")
        st.code(traceback.format_exc())
        st.stop()

# ── Engineer features ────────────────────────────────────────
try:
    df_eng = engineer_features(df_raw)
except Exception:
    st.error("❌ Feature engineering failed:")
    st.code(traceback.format_exc())
    st.stop()

# ── Verify 33-column shape ───────────────────────────────────
try:
    X_live, df_clean = verify_feature_shape(df_eng)
except ValueError as exc:
    st.error(f"❌ **Feature shape check failed:**\n\n{exc}")
    with st.expander("🔬 Feature Diagnostic", expanded=True):
        rows = []
        for col in FEATURE_COLS:
            if col in df_eng.columns:
                n_ok  = int(df_eng[col].notna().sum())
                last  = df_eng[col].dropna().iloc[-1] if n_ok else float("nan")
                rows.append({"Feature": col,
                              "Non-NaN rows": n_ok,
                              "Latest": f"{last:.4f}" if not np.isnan(last) else "NaN ⚠️",
                              "Status": "✅" if n_ok >= 1 else "❌"})
            else:
                rows.append({"Feature": col, "Non-NaN rows": 0,
                              "Latest": "MISSING", "Status": "❌ ABSENT"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.stop()
except Exception:
    st.error("❌ Unexpected error in verify_feature_shape():")
    st.code(traceback.format_exc())
    st.stop()


# ══════════════════════════════════════════════════════════════
# SECTION 3 — LIVE METRICS DISPLAY  (before the chart)
# Shows: Last Price · Change ($) · Change (%) · Open · Range · Volume
# Adds a 52-week high/low row for context.
# ══════════════════════════════════════════════════════════════
st.markdown("### 💹 LIVE MARKET SNAPSHOT")

try:
    latest      = df_raw.iloc[-1]
    prev        = df_raw.iloc[-2]
    cur_price   = float(latest["Close"])
    prev_price  = float(prev["Close"])
    chg_dollar  = cur_price - prev_price
    chg_pct     = chg_dollar / prev_price * 100
    day_open    = float(latest["Open"])
    day_high    = float(latest["High"])
    day_low     = float(latest["Low"])
    day_vol     = float(latest["Volume"])
    last_date   = df_raw.index[-1].strftime("%b %d, %Y")

    # 52-week high / low from full fetched history
    wk52_high = float(df_raw["High"].max())
    wk52_low  = float(df_raw["Low"].min())

    # Row 1 — primary metrics
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)

    r1c1.metric(
        label="💰 Current Price",
        value=f"${cur_price:.2f}",
        delta=f"{chg_dollar:+.2f}  ({chg_pct:+.2f}%)",
        delta_color="normal",
        help=f"Closing price as of {last_date}",
    )
    r1c2.metric(
        label="📈 Price Change ($)",
        value=f"${chg_dollar:+.2f}",
        delta=f"{chg_pct:+.2f}%",
        delta_color="normal",
    )
    r1c3.metric(
        label="📊 Change (%)",
        value=f"{chg_pct:+.2f}%",
        delta="vs prev close",
        delta_color="off",
    )
    r1c4.metric(
        label="📦 Volume",
        value=f"{day_vol/1e6:.2f}M",
        delta=f"Avg: {float(df_raw['Volume'].mean())/1e6:.2f}M",
        delta_color="off",
    )

    # Row 2 — secondary metrics
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.metric("📂 Open",        f"${day_open:.2f}")
    r2c2.metric("🔺 Day High",    f"${day_high:.2f}")
    r2c3.metric("🔻 Day Low",     f"${day_low:.2f}")
    r2c4.metric("📅 Last Updated", last_date)

    # Row 3 — 52-week context bar
    r3c1, r3c2, r3c3 = st.columns([1, 2, 1])
    r3c1.metric("52-wk Low",  f"${wk52_low:.2f}")
    r3c3.metric("52-wk High", f"${wk52_high:.2f}")

    with r3c2:
        pct_in_range = (cur_price - wk52_low) / max(wk52_high - wk52_low, 1e-9)
        st.markdown(
            f"""
            <div style='margin-top:8px'>
              <div style='font-size:10px;color:#3d8bcd;letter-spacing:1px;
                          margin-bottom:4px'>52-WEEK RANGE POSITION</div>
              <div style='background:#1e2d4a;border-radius:6px;height:10px;
                          overflow:hidden'>
                <div style='width:{pct_in_range*100:.1f}%;height:100%;
                            background:linear-gradient(90deg,#00d4ff,#00ff88);
                            border-radius:6px'></div>
              </div>
              <div style='text-align:right;font-size:10px;color:#00ff88;
                          margin-top:2px'>{pct_in_range*100:.1f}% of range</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

except Exception:
    st.warning("Metrics display failed:")
    st.code(traceback.format_exc())

st.divider()


# ══════════════════════════════════════════════════════════════
# PRICE CHART
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
        for sma, color in [
            ("SMA_20","#f59e0b"),
            ("SMA_50","#3b82f6"),
            ("SMA_200","#a855f7"),
        ]:
            if sma in df_plot.columns:
                fig.add_trace(go.Scatter(
                    x=df_plot.index, y=df_plot[sma],
                    name=sma, line=dict(color=color, width=1.2)
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
# PREDICTION
# ══════════════════════════════════════════════════════════════
st.markdown("### 🤖 AI PREDICTION")

if not artifacts["loaded"]:
    st.warning(
        "⚠️ **Model not loaded** — "
        "check the sidebar Model Info panel for the exact error."
    )
else:
    pipeline = artifacts["pipeline"]
    scaler   = artifacts["scaler"]

    # Shape guard — last line of defence before scaler.transform()
    n_expected, n_actual = scaler.n_features_in_, X_live.shape[1]
    if n_actual != n_expected:
        st.error(
            f"❌ **Shape mismatch blocked.**\n\n"
            f"| | |\n|---|---|\n"
            f"| App sends | **{n_actual}** features |\n"
            f"| Scaler expects | **{n_expected}** features |\n\n"
            "Re-train `stock_prediction_pipeline.py` with matching FEATURE_COLS."
        )
        with st.expander("Current FEATURE_COLS"):
            st.code("\n".join(f"{i+1:>2}. {c}" for i, c in enumerate(FEATURE_COLS)))
        st.stop()

    try:
        X_scaled        = scaler.transform(X_live)
        predicted_price = float(pipeline.predict(X_scaled)[0])
    except Exception:
        st.error("❌ Prediction failed:")
        st.code(traceback.format_exc())
        st.stop()

    # Prediction display card
    current_close = float(df_raw["Close"].iloc[-1])
    delta         = predicted_price - current_close
    pct           = delta / current_close * 100
    bullish       = delta > 0
    direction     = "📈 BULLISH" if bullish else "📉 BEARISH"
    d_color       = "#00ff88" if bullish else "#ef4444"

    st.markdown(f"""
    <div style='border:1px solid {d_color};border-radius:10px;padding:28px;
                background:rgba(0,0,0,.35);margin:10px 0 24px;text-align:center;
                box-shadow:0 0 24px {d_color}33'>
        <div style='font-family:Orbitron,sans-serif;font-size:13px;
                    color:{d_color};letter-spacing:4px;margin-bottom:10px'>
            PREDICTED NEXT-DAY CLOSE · {ticker}
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

    # Actual vs Predicted overlay
    try:
        n_hist   = min(120, len(df_clean))
        h_X      = df_clean[FEATURE_COLS].iloc[-n_hist:].values
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
            title=f"Actual vs Predicted — {ticker} (last {n_hist} sessions)",
            template="plotly_dark", paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1628",
            height=400, margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)
    except Exception:
        st.warning("Overlay chart failed:")
        st.code(traceback.format_exc())

    # Performance metrics
    st.markdown("### 🏆 MODEL PERFORMANCE (Test Set)")
    meta = artifacts["metadata"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE",      f"${meta.get('test_mae','N/A')}")
    m2.metric("RMSE",     f"${meta.get('test_rmse','N/A')}")
    m3.metric("R² Score", f"{meta.get('test_r2','N/A')}")
    m4.metric("MAPE",     f"{meta.get('test_mape','N/A')}%")

    # Debug expander
    with st.expander("🔬 Live Feature Values Sent to Scaler", expanded=False):
        dbg = (
            pd.DataFrame(X_live, columns=FEATURE_COLS)
            .T.rename(columns={0: "Live Value"})
        )
        dbg["Live Value"] = dbg["Live Value"].map(lambda x: f"{x:.6f}")
        st.dataframe(dbg, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS EXPANDER
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
# RECENT DATA TABLE
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


# ══════════════════════════════════════════════════════════════
# SECTION 4 — AUTO-REFRESH  (must be the very last block)
#
# Why last?
#   st.rerun() interrupts execution immediately.
#   If it were placed earlier, the page would restart before
#   any charts or predictions rendered — effectively blank screen.
#
# Why time.sleep() before st.rerun()?
#   Streamlit re-runs the entire script on every rerun().
#   Sleeping here holds the already-rendered page visible for
#   `refresh_secs` seconds before the next cycle begins.
#   The @st.cache_data(ttl=60) on get_stock_data() ensures
#   yfinance is only called when the cache actually expires.
# ══════════════════════════════════════════════════════════════
if live_tracking:
    # Show a visible countdown so the user knows the app is alive
    countdown_slot = st.empty()
    for remaining in range(refresh_secs, 0, -1):
        countdown_slot.markdown(
            f"<div style='text-align:center;color:#3d5a80;font-size:11px;"
            f"letter-spacing:2px'>🔄 REFRESHING IN {remaining}s</div>",
            unsafe_allow_html=True,
        )
        time.sleep(1)

    countdown_slot.empty()
    st.rerun()
