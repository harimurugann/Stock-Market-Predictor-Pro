# ==============================================================
# 
# Header Display
st.markdown(f"<h1 style='text-align:center;'>STOCKSIGHT AI: {ticker}</h1>", unsafe_allow_html=True)

# LIVE METRICS DISPLAY (Requested before the chart)
latest = df_raw.iloc[-1]
prev = df_raw.iloc[-2]
chg = float(latest["Close"]) - float(prev["Close"])
pct = (chg / float(prev["Close"])) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Price", f"${float(latest['Close']):.2f}", f"{chg:+.2f} ({pct:+.2f}%)")
c2.metric("Day High", f"${float(latest['High']):.2f}")
c3.metric("Day Low", f"${float(latest['Low']):.2f}")
c4.metric("Volume", f"{float(latest['Volume'])/1e6:.2f}M")

st.divider()

# CHART SECTION
st.markdown("### 📉 PRICE CHART")
fig = go.Figure(data=[go.Candlestick(x=df_raw.index, open=df_raw['Open'], high=df_raw['High'], low=df_raw['Low'], close=df_raw['Close'], name='OHLC')])
fig.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

# AI PREDICTION SECTION
st.markdown("### 🤖 AI PREDICTION")
if artifacts["loaded"] and X_live is not None:
    X_scaled = artifacts["scaler"].transform(X_live)
    pred = float(artifacts["pipeline"].predict(X_scaled)[0])
    
    current_val = float(df_raw["Close"].iloc[-1])
    diff = pred - current_val
    color = "#00ff88" if diff > 0 else "#ef4444"
    
    st.markdown(f"""
    <div style='border:1px solid {color}; border-radius:10px; padding:20px; text-align:center; background:rgba(0,0,0,0.3);'>
        <h4 style='color:{color}; margin:0;'>PREDICTED NEXT CLOSE</h4>
        <h1 style='color:{color}; font-size:3rem;'>${pred:.2f}</h1>
        <p style='color:#888;'>Direction: {"BULLISH" if diff > 0 else "BEARISH"} | Change: {diff:+.2f} ({(diff/current_val)*100:+.2f}%)</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.error("Model artifacts could not be loaded. Please check /model folder.")

# ──────────────────────────────────────────────────────────────
# 7. AUTO-REFRESH LOGIC (At the very end)
# ──────────────────────────────────────────────────────────────
if live_track:
    time.sleep(60)
    st.rerun()
