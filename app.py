"""
=============================================================================
  StockSight AI — Upgraded Streamlit App
  Features : Dropdown ticker, live refresh, robust yfinance, dark UI
  Model    : model/stock_pipeline_compressed.sav
=============================================================================
"""

import time
import warnings
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSight AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Dark Theme CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global background ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
}
[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #30363d;
}

/* ── Sidebar headings ── */
.sidebar-title {
    color: #39d353;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}

/* ── Main header ── */
.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #39d353;
    letter-spacing: 0.04em;
}
.main-subtitle {
    color: #8b949e;
    font-size: 0.95rem;
    margin-top: -0.5rem;
    margin-bottom: 1.2rem;
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    flex: 1;
    min-width: 130px;
}
.metric-label {
    font-size: 0.72rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.25rem;
}
.metric-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: #e6edf3;
}
.metric-delta-up   { color: #39d353; font-size: 0.85rem; font-weight: 600; }
.metric-delta-down { color: #f85149; font-size: 0.85rem; font-weight: 600; }
.metric-neutral    { color: #8b949e; font-size: 0.85rem; }

/* ── Live badge ── */
.live-badge {
    display: inline-block;
    background: #1f6feb;
    color: #ffffff;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 0.15rem 0.55rem;
    border-radius: 20px;
    letter-spacing: 0.08em;
    vertical-align: middle;
    margin-left: 0.5rem;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%   { opacity: 1; }
    50%  { opacity: 0.55; }
    100% { opacity: 1; }
}

/* ── Section header ── */
.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #39d353;
    letter-spacing: 0.06em;
    margin-top: 1.2rem;
    margin-bottom: 0.4rem;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.3rem;
}

/* ── Prediction banner ── */
.pred-up {
    background: linear-gradient(90deg, #0d4a1f 0%, #1a6b2e 100%);
    border: 1px solid #39d353;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    font-size: 1.4rem;
    font-weight: 800;
    color: #39d353;
    text-align: center;
}
.pred-down {
    background: linear-gradient(90deg, #4a0d0d 0%, #6b1a1a 100%);
    border: 1px solid #f85149;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    font-size: 1.4rem;
    font-weight: 800;
    color: #f85149;
    text-align: center;
}

/* ── Stray white backgrounds ── */
[data-testid="metric-container"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    padding: 0.6rem 0.8rem !important;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stSlider"] label { color: #c9d1d9 !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.55rem 1.4rem !important;
    width: 100% !important;
    letter-spacing: 0.04em !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Checkboxes & dividers ── */
[data-testid="stCheckbox"] span { color: #c9d1d9 !important; }
hr { border-color: #21262d !important; }
</style>
""", unsafe_allow_html=True)


# ─── Constants ────────────────────────────────────────────────────────────────
POPULAR_TICKERS = {
    "🇺🇸  Apple (AAPL)"          : "AAPL",
    "🇺🇸  NVIDIA (NVDA)"         : "NVDA",
    "🇺🇸  Tesla (TSLA)"          : "TSLA",
    "🇺🇸  Microsoft (MSFT)"      : "MSFT",
    "🇺🇸  Amazon (AMZN)"         : "AMZN",
    "🇺🇸  Alphabet (GOOGL)"      : "GOOGL",
    "🇺🇸  Meta (META)"           : "META",
    "🇺🇸  S&P 500 ETF (SPY)"     : "SPY",
    "🇮🇳  Reliance (RELIANCE.NS)": "RELIANCE.NS",
    "🇮🇳  TCS (TCS.NS)"          : "TCS.NS",
    "🇮🇳  Infosys (INFY.NS)"     : "INFY.NS",
    "🇮🇳  SBI (SBIN.NS)"         : "SBIN.NS",
    "🇮🇳  HDFC Bank (HDFCBANK.NS)":"HDFCBANK.NS",
    "🇮🇳  Wipro (WIPRO.NS)"      : "WIPRO.NS",
    "✏️  Other (type manually)"  : "__OTHER__",
}

MODEL_PATH   = "model/stock_pipeline_compressed.sav"
REFRESH_SECS = 30   # live refresh interval in seconds


# ─── Helpers ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the saved pipeline. Cached so it is not reloaded on every refresh."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def fetch_data(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Robust yfinance download that avoids empty-DataFrame errors.
    Falls back from Ticker.history() to yf.download() automatically.
    """
    try:
        # Method 1 — Ticker.history (preferred, avoids multi-level column issues)
        tk  = yf.Ticker(ticker)
        df  = tk.history(period=period, interval=interval, auto_adjust=True)
        if df is not None and not df.empty and len(df) > 5:
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df[["Open", "High", "Low", "Close", "Volume"]].copy()
    except Exception:
        pass

    try:
        # Method 2 — yf.download fallback
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df is not None and not df.empty:
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df[["Open", "High", "Low", "Close", "Volume"]].copy()
    except Exception:
        pass

    return pd.DataFrame()


def fetch_live_quote(ticker: str) -> dict:
    """
    Fetch the latest quote (price, open, day range, volume).
    Returns a dict with safe fallback values on failure.
    """
    defaults = {
        "price": None, "open": None, "prev_close": None,
        "day_low": None, "day_high": None,
        "volume": None, "change": None, "change_pct": None,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        tk   = yf.Ticker(ticker)
        info = tk.fast_info           # lightweight endpoint
        price      = getattr(info, "last_price",       None)
        prev_close = getattr(info, "previous_close",   None)
        open_      = getattr(info, "open",              None)
        day_low    = getattr(info, "day_low",           None)
        day_high   = getattr(info, "day_high",          None)
        volume     = getattr(info, "last_volume",       None)

        # Derive change metrics
        if price and prev_close and prev_close != 0:
            change     = price - prev_close
            change_pct = (change / prev_close) * 100
        else:
            change = change_pct = None

        return {
            "price"       : price,
            "open"        : open_,
            "prev_close"  : prev_close,
            "day_low"     : day_low,
            "day_high"    : day_high,
            "volume"      : volume,
            "change"      : change,
            "change_pct"  : change_pct,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception:
        return defaults


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MA20, MA50 and any other features expected by the model."""
    df = df.copy()
    df["MA20"]   = df["Close"].rolling(20).mean()
    df["MA50"]   = df["Close"].rolling(50).mean()
    df["Return"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    return df


def fmt_volume(v):
    """Human-readable volume string."""
    if v is None:
        return "N/A"
    if v >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v/1_000:.1f}K"
    return str(int(v))


def delta_html(change, change_pct):
    """Return colour-coded delta HTML snippet."""
    if change is None:
        return '<span class="metric-neutral">— N/A</span>'
    arrow = "▲" if change >= 0 else "▼"
    css   = "metric-delta-up" if change >= 0 else "metric-delta-down"
    sign  = "+" if change >= 0 else ""
    return (f'<span class="{css}">'
            f'{arrow} {sign}{change:.2f} ({sign}{change_pct:.2f}%)'
            f'</span>')


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ Configuration</div>',
                unsafe_allow_html=True)

    # ── Ticker dropdown ──────────────────────────────────────────────────────
    selected_label = st.selectbox(
        "Ticker Symbol",
        options=list(POPULAR_TICKERS.keys()),
        index=2,           # default → TSLA
        help="Choose a stock or select 'Other' to type a custom ticker.",
    )
    resolved_ticker = POPULAR_TICKERS[selected_label]

    if resolved_ticker == "__OTHER__":
        custom_input = st.text_input(
            "Enter custom ticker",
            value="",
            max_chars=15,
            placeholder="e.g. BAJFINANCE.NS",
        ).strip().upper()
        ticker = custom_input if custom_input else "TSLA"
    else:
        ticker = resolved_ticker

    st.caption(f"Active ticker: **{ticker}**")

    st.divider()

    # ── Chart settings ───────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-title">📊 Chart Settings</div>',
                unsafe_allow_html=True)
    show_ma     = st.checkbox("Show Moving Averages", value=True)
    show_bb     = st.checkbox("Show Bollinger Bands",  value=True)
    show_volume = st.checkbox("Show Volume",           value=True)

    chart_period = st.select_slider(
        "History Period",
        options=["1mo", "3mo", "6mo", "1y", "2y"],
        value="6mo",
    )

    st.divider()

    # ── Live refresh ─────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-title">🔄 Live Refresh</div>',
                unsafe_allow_html=True)
    auto_refresh = st.toggle("Auto-Refresh Price", value=False)
    refresh_rate = st.slider(
        "Refresh every (seconds)",
        min_value=10, max_value=120, value=REFRESH_SECS, step=5,
        disabled=not auto_refresh,
    )

    st.divider()

    # ── Run prediction button ─────────────────────────────────────────────────
    run_prediction = st.button("🚀  RUN PREDICTION", use_container_width=True)


# ─── Main Panel ───────────────────────────────────────────────────────────────
st.markdown('<span class="main-title">📈 StockSight AI</span>'
            '<span class="live-badge">LIVE</span>', unsafe_allow_html=True)
st.markdown(f'<div class="main-subtitle">Real-time market intelligence • {ticker}</div>',
            unsafe_allow_html=True)


# ── Live quote container (refreshes without full reload) ──────────────────────
quote_container  = st.empty()
chart_container  = st.container()
pred_container   = st.container()


def render_quote(q: dict):
    """Render the top metric bar inside the empty container."""
    price_str = f"${q['price']:,.2f}"  if q["price"]      else "N/A"
    open_str  = f"${q['open']:,.2f}"   if q["open"]       else "N/A"
    range_str = (f"${q['day_low']:,.2f} – ${q['day_high']:,.2f}"
                 if q["day_low"] and q["day_high"] else "N/A")
    vol_str   = fmt_volume(q["volume"])
    delta     = delta_html(q["change"], q["change_pct"])
    ts        = q["last_updated"]
    pc_str    = f"${q['prev_close']:,.2f}" if q["prev_close"] else "N/A"

    html = f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-label">💰 Close / Last</div>
        <div class="metric-value">{price_str}</div>
        <div>{delta}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">🟡 Open</div>
        <div class="metric-value">{open_str}</div>
        <div class="metric-neutral">Prev Close {pc_str}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">📏 Day Range</div>
        <div class="metric-value" style="font-size:1.15rem">{range_str}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">📦 Volume</div>
        <div class="metric-value">{vol_str}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">🕐 Last Updated</div>
        <div class="metric-value" style="font-size:1.05rem">{ts}</div>
        <div class="metric-neutral">Auto-refresh: {'ON' if auto_refresh else 'OFF'}</div>
      </div>
    </div>
    """
    quote_container.markdown(html, unsafe_allow_html=True)


# ── First quote fetch ──────────────────────────────────────────────────────────
with st.spinner(f"Fetching live quote for {ticker} …"):
    quote = fetch_live_quote(ticker)
render_quote(quote)


# ── Historical data + chart ────────────────────────────────────────────────────
with st.spinner(f"Loading {chart_period} price history …"):
    df_raw = fetch_data(ticker, period=chart_period)

with chart_container:
    st.markdown('<div class="section-header">📊 PRICE CHART</div>',
                unsafe_allow_html=True)

    if df_raw.empty:
        st.error(f"❌  Could not fetch data for **{ticker}**. "
                 "Please check the ticker symbol and try again.")
    else:
        df = add_features(df_raw.copy())

        # ── Bollinger Bands (20-period, ±2σ) ──────────────────────────────
        if show_bb and len(df) >= 20:
            rolling       = df["Close"].rolling(20)
            df["BB_mid"]  = rolling.mean()
            df["BB_upper"]= df["BB_mid"] + 2 * rolling.std()
            df["BB_lower"]= df["BB_mid"] - 2 * rolling.std()

        # ── Build Plotly chart ─────────────────────────────────────────────
        row_heights = [0.75, 0.25] if show_volume else [1.0]
        n_rows = 2 if show_volume else 1

        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=n_rows, cols=1, shared_xaxes=True,
            row_heights=row_heights,
            vertical_spacing=0.03,
        )

        # OHLC candlesticks
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="OHLC",
            increasing_line_color="#39d353",
            decreasing_line_color="#f85149",
            increasing_fillcolor="#39d353",
            decreasing_fillcolor="#f85149",
        ), row=1, col=1)

        # Moving averages
        if show_ma:
            for col, colour, dash in [
                ("MA20", "#58a6ff", "solid"),
                ("MA50", "#e3b341", "solid"),
            ]:
                if col in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df[col], name=col,
                        line=dict(color=colour, width=1.5, dash=dash),
                        opacity=0.85,
                    ), row=1, col=1)

            # SMA 200 (if enough data)
            if len(df_raw) >= 200:
                df["MA200"] = df_raw["Close"].rolling(200).mean().loc[df.index]
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["MA200"], name="SMA_200",
                    line=dict(color="#bc8cff", width=1.5),
                    opacity=0.8,
                ), row=1, col=1)

        # Bollinger Bands
        if show_bb and "BB_upper" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["BB_upper"], name="BB Upper",
                line=dict(color="#58a6ff", width=1, dash="dash"), opacity=0.7,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=df["BB_lower"], name="BB Lower",
                line=dict(color="#58a6ff", width=1, dash="dash"), opacity=0.7,
                fill="tonexty",
                fillcolor="rgba(88,166,255,0.05)",
            ), row=1, col=1)

        # Volume bars
        if show_volume:
            colours = [
                "#39d353" if c >= o else "#f85149"
                for c, o in zip(df["Close"], df["Open"])
            ]
            fig.add_trace(go.Bar(
                x=df.index, y=df["Volume"], name="Volume",
                marker_color=colours, opacity=0.65,
            ), row=2, col=1)

        # Layout
        fig.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#c9d1d9", size=12),
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(
                bgcolor="rgba(22,27,34,0.8)",
                bordercolor="#30363d",
                borderwidth=1,
                font=dict(size=11),
                orientation="v",
                x=1.01, y=1,
            ),
            xaxis_rangeslider_visible=False,
            height=520 if show_volume else 420,
            hovermode="x unified",
        )
        fig.update_xaxes(
            gridcolor="#21262d", zeroline=False,
            showspikes=True, spikecolor="#58a6ff",
            spikemode="across", spikethickness=1,
        )
        fig.update_yaxes(gridcolor="#21262d", zeroline=False)

        st.plotly_chart(fig, use_container_width=True)


# ── Prediction ────────────────────────────────────────────────────────────────
with pred_container:
    st.markdown('<div class="section-header">🤖 AI PREDICTION</div>',
                unsafe_allow_html=True)

    if run_prediction:
        model = load_model()
        if model is None:
            st.error(f"❌  Model not found at `{MODEL_PATH}`. "
                     "Please ensure the file exists and re-run.")
        elif df_raw.empty:
            st.warning("⚠️  No price data available — cannot generate a prediction.")
        else:
            try:
                df_feat = add_features(df_raw.copy())
                # Use the last available row for prediction
                latest  = df_feat.iloc[[-1]][["Open", "High", "Low",
                                              "Close", "Volume",
                                              "MA20", "MA50"]]
                prediction  = model.predict(latest)[0]
                probability = model.predict_proba(latest)[0]
                prob_up     = probability[1] * 100

                if prediction == 1:
                    st.markdown(
                        f'<div class="pred-up">'
                        f'📈 Prediction: BULLISH &nbsp;|&nbsp; '
                        f'Confidence: {prob_up:.1f}%'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="pred-down">'
                        f'📉 Prediction: BEARISH &nbsp;|&nbsp; '
                        f'Confidence: {100-prob_up:.1f}%'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                # Probability bar
                st.markdown("")
                col_p1, col_p2 = st.columns(2)
                col_p1.metric("🟢 Bullish Probability", f"{prob_up:.1f}%")
                col_p2.metric("🔴 Bearish Probability", f"{100-prob_up:.1f}%")

                st.caption(
                    "⚠️ **Disclaimer**: This prediction is generated by an ML model "
                    "for educational purposes only. It is NOT financial advice."
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("Tip: Ensure the model was trained with features: "
                        "Open, High, Low, Close, Volume, MA20, MA50.")
    else:
        st.info("👈  Configure your ticker and click **RUN PREDICTION** to get the AI forecast.")


# ─── Auto-Refresh Logic ───────────────────────────────────────────────────────
# Uses st.rerun() after sleeping, inside a conditional block so it
# only activates when the toggle is ON — no infinite loop when OFF.
if auto_refresh:
    time.sleep(refresh_rate)
    # Refresh quote only (lightweight) before rerunning
    refreshed = fetch_live_quote(ticker)
    render_quote(refreshed)
    st.rerun()


# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "StockSight AI • Powered by yfinance & scikit-learn • "
    "For research & educational use only • Not financial advice"
)
