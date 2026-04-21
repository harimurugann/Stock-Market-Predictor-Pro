"""
=============================================================================
  StockSight AI — Streamlit Web Application
  Theme   : Neon-green dark (GitHub-dark palette)
  Model   : model/stock_pipeline_compressed.sav  (resolved from __file__)
=============================================================================
"""

import os
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSight AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. ROBUST PATH HANDLING
#    os.path.dirname(__file__) ensures the path resolves correctly both
#    locally and on Streamlit Cloud, regardless of the working directory.
# ─────────────────────────────────────────────────────────────────────────────
BASE_PATH  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "model", "stock_pipeline_compressed.sav")

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TICKER_OPTIONS = [
    "TSLA", "AAPL", "NVDA", "MSFT", "AMZN",
    "GOOGL", "META", "SPY",
    "RELIANCE.NS", "TCS.NS", "INFY.NS",
    "SBIN.NS", "HDFCBANK.NS", "WIPRO.NS",
    "Other (type manually)",
]

# MA-200 needs at least 200 rows; add 20 buffer for BB warm-up
MIN_ROWS_FOR_MA = 220

# ─────────────────────────────────────────────────────────────────────────────
# NEON-GREEN DARK THEME
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"]                   { background-color:#0d1117 !important; color:#e6edf3 !important; }
[data-testid="stSidebar"]               { background-color:#161b22 !important; border-right:1px solid #30363d; }
[data-testid="stSidebar"] *             { color:#c9d1d9 !important; }
h1,h2,h3,h4                             { color:#e6edf3 !important; }

.sb-heading {
    color:#39d353 !important; font-size:.88rem; font-weight:700;
    letter-spacing:.09em; text-transform:uppercase;
    margin:.8rem 0 .35rem 0;
}
.app-title { font-size:2.1rem; font-weight:900; color:#39d353; letter-spacing:.05em; line-height:1.1; }
.app-sub   { color:#8b949e; font-size:.9rem; margin-top:.1rem; margin-bottom:1rem; }

.live-badge {
    display:inline-block; background:#1f6feb; color:#fff;
    font-size:.65rem; font-weight:800; padding:.12rem .5rem;
    border-radius:20px; letter-spacing:.1em; vertical-align:middle;
    margin-left:.6rem; animation:blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.45} }

.kpi-row  { display:flex; gap:.75rem; flex-wrap:wrap; margin-bottom:1rem; }
.kpi-card { background:#161b22; border:1px solid #30363d; border-radius:10px;
            padding:.75rem 1.1rem; flex:1; min-width:120px; }
.kpi-lbl  { font-size:.68rem; color:#8b949e; text-transform:uppercase;
            letter-spacing:.07em; margin-bottom:.2rem; }
.kpi-val  { font-size:1.5rem; font-weight:700; color:#e6edf3; }
.kpi-up   { color:#39d353; font-size:.82rem; font-weight:600; }
.kpi-down { color:#f85149; font-size:.82rem; font-weight:600; }
.kpi-flat { color:#8b949e; font-size:.82rem; }

.sec-hdr {
    font-size:1.1rem; font-weight:700; color:#39d353;
    letter-spacing:.07em; border-bottom:1px solid #21262d;
    padding-bottom:.3rem; margin-top:1.1rem; margin-bottom:.5rem;
}

.bull-banner {
    background:linear-gradient(90deg,#0d4a1f,#1a6b2e);
    border:1px solid #39d353; border-radius:10px;
    padding:1rem 1.4rem; font-size:1.35rem; font-weight:800;
    color:#39d353; text-align:center; margin-top:.5rem;
}
.bear-banner {
    background:linear-gradient(90deg,#4a0d0d,#6b1a1a);
    border:1px solid #f85149; border-radius:10px;
    padding:1rem 1.4rem; font-size:1.35rem; font-weight:800;
    color:#f85149; text-align:center; margin-top:.5rem;
}

[data-testid="metric-container"] {
    background:#161b22 !important; border:1px solid #30363d !important;
    border-radius:10px !important; padding:.6rem .9rem !important;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stSlider"]    label { color:#c9d1d9 !important; }
div[data-testid="stCheckbox"]  span  { color:#c9d1d9 !important; }

.stButton > button {
    background:linear-gradient(135deg,#238636,#2ea043) !important;
    color:#fff !important; border:none !important;
    border-radius:8px !important; font-weight:700 !important;
    font-size:.95rem !important; padding:.55rem 1.2rem !important;
    width:100% !important; letter-spacing:.04em !important;
    transition:opacity .2s !important;
}
.stButton > button:hover { opacity:.82 !important; }
hr { border-color:#21262d !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """
    Load the trained sklearn pipeline once per session.
    Returns None (without raising) if the file is missing so the app
    can show a user-friendly error instead of crashing.
    """
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shared cleanup: flatten multi-level columns, strip timezone,
    keep only OHLCV columns.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[needed].copy()


def fetch_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch OHLCV history using a two-method fallback strategy.
    Method 1 — Ticker.history() : cleanest output
    Method 2 — yf.download()    : fallback, auto-flattens columns
    """
    # Method 1
    try:
        raw = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
        cleaned = _clean_df(raw)
        if not cleaned.empty and len(cleaned) > 5:
            return cleaned
    except Exception:
        pass

    # Method 2
    try:
        raw = yf.download(ticker, period=period, interval="1d",
                          auto_adjust=True, progress=False, threads=False)
        cleaned = _clean_df(raw)
        if not cleaned.empty:
            return cleaned
    except Exception:
        pass

    return pd.DataFrame()


def fetch_live_quote(ticker: str) -> dict:
    """
    Fetch the latest real-time quote via yfinance fast_info.
    Falls back to a 2-day history close if fast_info fails.
    All fields default to None so callers never see a KeyError.
    """
    q = dict(price=None, prev_close=None, open=None,
             day_low=None, day_high=None, volume=None,
             change=None, change_pct=None,
             updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    try:
        fi = yf.Ticker(ticker).fast_info
        q["price"]      = getattr(fi, "last_price",     None)
        q["prev_close"] = getattr(fi, "previous_close", None)
        q["open"]       = getattr(fi, "open",            None)
        q["day_low"]    = getattr(fi, "day_low",         None)
        q["day_high"]   = getattr(fi, "day_high",        None)
        q["volume"]     = getattr(fi, "last_volume",     None)
    except Exception:
        pass

    # Fallback price from history
    if q["price"] is None:
        try:
            tmp = yf.Ticker(ticker).history(period="2d", interval="1d", auto_adjust=True)
            if not tmp.empty:
                q["price"]      = float(tmp["Close"].iloc[-1])
                q["prev_close"] = float(tmp["Close"].iloc[-2]) if len(tmp) > 1 else None
        except Exception:
            pass

    # Derive delta
    p, pc = q["price"], q["prev_close"]
    if p and pc and pc != 0:
        q["change"]     = p - pc
        q["change_pct"] = (p - pc) / pc * 100

    return q


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append MA-20, MA-50, MA-200, Bollinger Bands (20-period ±2σ),
    and daily return to the DataFrame.
    Only rows without MA-20 and Return are dropped (MA-50/200 may be NaN
    early in the series and are still useful for charting).
    """
    df = df.copy()
    df["MA20"]   = df["Close"].rolling(20).mean()
    df["MA50"]   = df["Close"].rolling(50).mean()
    df["MA200"]  = df["Close"].rolling(200).mean()
    df["BB_mid"] = df["Close"].rolling(20).mean()
    df["BB_std"] = df["Close"].rolling(20).std()
    df["BB_up"]  = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_lo"]  = df["BB_mid"] - 2 * df["BB_std"]
    df["Return"] = df["Close"].pct_change()
    df.dropna(subset=["MA20", "Return"], inplace=True)
    return df


def fmt_vol(v) -> str:
    if v is None: return "N/A"
    v = int(v)
    if v >= 1_000_000: return f"{v/1_000_000:.2f}M"
    if v >= 1_000:     return f"{v/1_000:.1f}K"
    return str(v)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown('<p class="sb-heading">⚙️ Configuration</p>', unsafe_allow_html=True)

    # 2. Ticker dropdown with 'Other' option
    selected = st.selectbox(
        "Ticker Symbol",
        options=TICKER_OPTIONS,
        index=0,
        help="Pick a stock or choose 'Other' to type any ticker.",
    )
    if selected == "Other (type manually)":
        custom = st.text_input(
            "Enter ticker symbol", value="", max_chars=20,
            placeholder="e.g. BAJFINANCE.NS",
        ).strip().upper()
        TICKER = custom if custom else "TSLA"
    else:
        TICKER = selected

    st.caption(f"Active: **{TICKER}**")
    st.divider()

    st.markdown('<p class="sb-heading">📊 Chart Settings</p>', unsafe_allow_html=True)
    show_ma  = st.checkbox("Show Moving Averages", value=True)
    show_bb  = st.checkbox("Show Bollinger Bands",  value=True)
    show_vol = st.checkbox("Show Volume",           value=True)
    CHART_PERIOD = st.select_slider(
        "History Period",
        options=["1mo", "3mo", "6mo", "1y", "2y"],
        value="6mo",
    )
    st.divider()

    st.markdown('<p class="sb-heading">🔄 Live Refresh</p>', unsafe_allow_html=True)
    auto_refresh  = st.toggle("Auto-Refresh Price", value=False)
    refresh_every = st.slider(
        "Refresh every (seconds)", min_value=10, max_value=120,
        value=30, step=5, disabled=not auto_refresh,
    )
    st.divider()

    run_btn = st.button("🚀  RUN PREDICTION", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PANEL — Title
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f'<span class="app-title">📈 StockSight AI</span>'
    f'<span class="live-badge">LIVE</span>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<p class="app-sub">Real-time market intelligence • {TICKER}</p>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. LIVE PRICE TRACKING — st.empty() container for non-full-page refresh
# ─────────────────────────────────────────────────────────────────────────────
quote_slot = st.empty()


def render_quote(q: dict) -> None:
    """Render KPI bar into the pre-allocated empty container."""
    def _fp(v): return f"${v:,.2f}" if v else "N/A"

    range_s = (f"${q['day_low']:,.2f} – ${q['day_high']:,.2f}"
               if q["day_low"] and q["day_high"] else "N/A")

    if q["change"] is None:
        delta = '<span class="kpi-flat">— N/A</span>'
    elif q["change"] >= 0:
        delta = (f'<span class="kpi-up">▲ +{q["change"]:.2f} '
                 f'(+{q["change_pct"]:.2f}%)</span>')
    else:
        delta = (f'<span class="kpi-down">▼ {q["change"]:.2f} '
                 f'({q["change_pct"]:.2f}%)</span>')

    quote_slot.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-lbl">💰 Close / Last</div>
        <div class="kpi-val">{_fp(q['price'])}</div>
        <div>{delta}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-lbl">🟡 Open</div>
        <div class="kpi-val">{_fp(q['open'])}</div>
        <div class="kpi-flat">Prev Close {_fp(q['prev_close'])}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-lbl">📏 Day Range</div>
        <div class="kpi-val" style="font-size:1.05rem">{range_s}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-lbl">📦 Volume</div>
        <div class="kpi-val">{fmt_vol(q['volume'])}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-lbl">🕐 Last Updated</div>
        <div class="kpi-val" style="font-size:.95rem">{q['updated']}</div>
        <div class="kpi-flat">Auto-refresh {'ON ✅' if auto_refresh else 'OFF'}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


with st.spinner(f"Fetching live quote for {TICKER} …"):
    quote = fetch_live_quote(TICKER)
render_quote(quote)

# ─────────────────────────────────────────────────────────────────────────────
# 4. ENHANCED CHART — Candlestick + MAs + BB + Volume
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📊 PRICE CHART</div>', unsafe_allow_html=True)

with st.spinner(f"Loading {CHART_PERIOD} history for {TICKER} …"):
    df_hist = fetch_history(TICKER, period=CHART_PERIOD)

if df_hist.empty:
    st.error(f"❌  Could not fetch data for **{TICKER}**. Please verify the ticker symbol.")
else:
    df_chart = add_indicators(df_hist.copy())

    n_rows      = 2 if show_vol else 1
    row_heights = [0.73, 0.27] if show_vol else [1.0]

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.03,
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df_chart.index,
        open=df_chart["Open"], high=df_chart["High"],
        low=df_chart["Low"],   close=df_chart["Close"],
        name="OHLC",
        increasing_line_color="#39d353", decreasing_line_color="#f85149",
        increasing_fillcolor="#39d353",  decreasing_fillcolor="#f85149",
        line=dict(width=1),
    ), row=1, col=1)

    # Moving averages (SMA_20, SMA_50, SMA_200)
    if show_ma:
        for col, colour in [("MA20","#58a6ff"), ("MA50","#e3b341"), ("MA200","#bc8cff")]:
            if col in df_chart.columns and df_chart[col].notna().any():
                fig.add_trace(go.Scatter(
                    x=df_chart.index, y=df_chart[col],
                    name=col, line=dict(color=colour, width=1.5),
                    opacity=0.85,
                ), row=1, col=1)

    # Bollinger Bands
    if show_bb and "BB_up" in df_chart.columns:
        fig.add_trace(go.Scatter(
            x=df_chart.index, y=df_chart["BB_up"], name="BB Upper",
            line=dict(color="#58a6ff", width=1, dash="dash"), opacity=0.7,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_chart.index, y=df_chart["BB_lo"], name="BB Lower",
            line=dict(color="#58a6ff", width=1, dash="dash"),
            fill="tonexty", fillcolor="rgba(88,166,255,0.05)", opacity=0.7,
        ), row=1, col=1)

    # Volume bars
    if show_vol:
        colours = [
            "#39d353" if c >= o else "#f85149"
            for c, o in zip(df_chart["Close"], df_chart["Open"])
        ]
        fig.add_trace(go.Bar(
            x=df_chart.index, y=df_chart["Volume"],
            name="Volume", marker_color=colours, opacity=0.65,
        ), row=2, col=1)

    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9", size=11),
        margin=dict(l=8, r=8, t=8, b=8),
        legend=dict(
            bgcolor="rgba(22,27,34,0.85)", bordercolor="#30363d",
            borderwidth=1, font=dict(size=10),
            orientation="v", x=1.01, y=1,
        ),
        xaxis_rangeslider_visible=False,
        height=510 if show_vol else 400,
        hovermode="x unified",
    )
    fig.update_xaxes(
        gridcolor="#21262d", zeroline=False,
        showspikes=True, spikecolor="#58a6ff",
        spikemode="across", spikethickness=1,
        tickfont=dict(color="#8b949e"),
    )
    fig.update_yaxes(gridcolor="#21262d", zeroline=False, tickfont=dict(color="#8b949e"))

    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PREDICTION LOGIC
#    Always fetches 2y of data so MA-20/50/200 are fully warmed up
#    before the latest row is passed to the model.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">🤖 AI PREDICTION</div>', unsafe_allow_html=True)

if run_btn:
    model = load_model(MODEL_PATH)

    if model is None:
        st.error(
            f"❌  Model not found at `{MODEL_PATH}`.\n\n"
            "**Checklist for Streamlit Cloud:**\n"
            "1. Confirm `model/stock_pipeline_compressed.sav` is committed to your repo.\n"
            "2. Ensure the file is **not** in `.gitignore`.\n"
            "3. Verify `requirements.txt` includes `scikit-learn` and `joblib`.\n"
            "4. Redeploy the app after pushing the model file."
        )
    else:
        # Fetch 2 years so all MAs are warm by the final row
        with st.spinner(f"Fetching 2-year history for prediction ({TICKER}) …"):
            df_p_raw = fetch_history(TICKER, period="2y")

        if df_p_raw.empty:
            st.warning("⚠️  No historical data — cannot generate a prediction.")
        else:
            df_p = add_indicators(df_p_raw.copy())
            # For the prediction row itself, we need MA50 at minimum
            df_p.dropna(subset=["MA20", "MA50"], inplace=True)

            if df_p.empty:
                st.warning("⚠️  Not enough data after computing indicators.")
            else:
                if len(df_p_raw) < MIN_ROWS_FOR_MA:
                    st.warning(
                        f"Only {len(df_p_raw)} rows available; MA-200 may not be "
                        "fully warmed. Prediction uses MA-20 and MA-50 only."
                    )
                try:
                    feature_cols = ["Open", "High", "Low", "Close",
                                    "Volume", "MA20", "MA50"]
                    latest = df_p.iloc[[-1]][feature_cols]

                    prediction  = model.predict(latest)[0]
                    probability = model.predict_proba(latest)[0]
                    prob_up     = probability[1] * 100 if len(probability) > 1 else 50.0

                    if prediction == 1:
                        st.markdown(
                            f'<div class="bull-banner">📈 Prediction: BULLISH &nbsp;|&nbsp;'
                            f' Confidence: {prob_up:.1f}%</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="bear-banner">📉 Prediction: BEARISH &nbsp;|&nbsp;'
                            f' Confidence: {100-prob_up:.1f}%</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown("")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("🟢 Bullish Probability", f"{prob_up:.1f}%")
                    c2.metric("🔴 Bearish Probability", f"{100-prob_up:.1f}%")
                    c3.metric("📅 Signal Date", df_p.index[-1].strftime("%Y-%m-%d"))

                    with st.expander("🔍 Features fed to the model"):
                        row = latest.copy()
                        row.index = ["Latest candle"]
                        st.dataframe(row.style.format("{:.4f}"), use_container_width=True)

                    st.caption(
                        "⚠️ **Disclaimer** — For educational & research use only. "
                        "Not financial advice."
                    )

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.info(
                        "Ensure the pipeline was trained with features: "
                        "`Open, High, Low, Close, Volume, MA20, MA50`"
                    )
else:
    st.info("👈  Select a ticker and click **RUN PREDICTION** to get the AI forecast.")


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-REFRESH  (only fires when the sidebar toggle is ON)
# ─────────────────────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_every)
    render_quote(fetch_live_quote(TICKER))
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "StockSight AI  •  Powered by yfinance & scikit-learn  •  "
    "For research & educational use only  •  Not financial advice"
)
