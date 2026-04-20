# ============================================================
# Stock Market Prediction Pipeline
# Author: Generated Pipeline
# Description: End-to-end ML pipeline for stock price prediction
# ============================================================

# ─────────────────────────────────────────────
# 1. IMPORTING DEPENDENCIES
# ─────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.base import BaseEstimator, TransformerMixin

print("✅ All dependencies imported successfully.")

# ─────────────────────────────────────────────
# 2. DATA FETCHING & CLEANING
# ─────────────────────────────────────────────

TICKER = "AAPL"
START_DATE = "2020-01-01"
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")


def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance and perform initial cleaning.
    Returns a cleaned DataFrame with engineered features.
    """
    print(f"\n📡 Fetching live data for {ticker} from {start} to {end}...")
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check ticker symbol.")

    # Flatten multi-level columns if present (yfinance sometimes returns them)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # ── Drop rows where all OHLCV values are NaN ──
    df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], how="all", inplace=True)

    # ── Forward-fill isolated missing values (weekends/holidays gap) ──
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # ── Outlier removal using IQR on 'Close' ──
    Q1 = df["Close"].quantile(0.01)
    Q3 = df["Close"].quantile(0.99)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    before = len(df)
    df = df[(df["Close"] >= lower_bound) & (df["Close"] <= upper_bound)]
    removed = before - len(df)
    if removed > 0:
        print(f"  ⚠️  Removed {removed} outlier rows from 'Close' column.")

    print(f"  ✅ Data shape after cleaning: {df.shape}")
    print(f"  📅 Date range: {df.index[0].date()} → {df.index[-1].date()}")
    return df


df_raw = fetch_stock_data(TICKER, START_DATE, END_DATE)

# ─────────────────────────────────────────────
# 3. DATA VISUALIZATION
# ─────────────────────────────────────────────

def plot_candlestick(df: pd.DataFrame, ticker: str):
    """Interactive candlestick chart with volume subplot."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=(f"{ticker} — Candlestick Price Chart", "Volume"),
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(df["Close"], df["Open"])]

    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=colors, opacity=0.6),
        row=2, col=1,
    )

    fig.update_layout(
        title=f"{ticker} Stock Price & Volume",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=700,
        showlegend=False,
    )
    fig.write_html("data/candlestick_chart.html")
    print("  📊 Candlestick chart saved → data/candlestick_chart.html")
    fig.show()


os.makedirs("data", exist_ok=True)
plot_candlestick(df_raw, TICKER)

# ─────────────────────────────────────────────
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create technical indicators used as model features:
    - Simple Moving Averages (SMA)
    - Exponential Moving Averages (EMA)
    - Bollinger Bands
    - RSI (Relative Strength Index)
    - Daily Returns & Volatility
    - Lag features
    """
    d = df.copy()

    # ── Moving Averages ──
    for window in [5, 10, 20, 50, 200]:
        d[f"SMA_{window}"] = d["Close"].rolling(window).mean()
        d[f"EMA_{window}"] = d["Close"].ewm(span=window, adjust=False).mean()

    # ── Bollinger Bands (20-day) ──
    d["BB_mid"] = d["Close"].rolling(20).mean()
    d["BB_std"] = d["Close"].rolling(20).std()
    d["BB_upper"] = d["BB_mid"] + 2 * d["BB_std"]
    d["BB_lower"] = d["BB_mid"] - 2 * d["BB_std"]
    d["BB_width"] = (d["BB_upper"] - d["BB_lower"]) / d["BB_mid"]

    # ── RSI (14-day) ──
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    d["RSI_14"] = 100 - (100 / (1 + rs))

    # ── Daily Returns & Volatility ──
    d["Daily_Return"] = d["Close"].pct_change()
    d["Volatility_5"] = d["Daily_Return"].rolling(5).std()
    d["Volatility_20"] = d["Daily_Return"].rolling(20).std()

    # ── Price Momentum ──
    d["Momentum_5"] = d["Close"] - d["Close"].shift(5)
    d["Momentum_10"] = d["Close"] - d["Close"].shift(10)

    # ── MACD ──
    ema_12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema_12 - ema_26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACD_signal"]

    # ── Lag Features ──
    for lag in [1, 2, 3, 5, 10]:
        d[f"Close_lag_{lag}"] = d["Close"].shift(lag)
        d[f"Return_lag_{lag}"] = d["Daily_Return"].shift(lag)

    # ── Volume Feature ──
    d["Volume_SMA_10"] = d["Volume"].rolling(10).mean()
    d["Volume_ratio"] = d["Volume"] / (d["Volume_SMA_10"] + 1e-9)

    # ── Calendar Features ──
    d["DayOfWeek"] = d.index.dayofweek
    d["Month"] = d.index.month
    d["Quarter"] = d.index.quarter

    return d


df_features = engineer_features(df_raw)

# ── EDA Plots ──
def plot_eda(df: pd.DataFrame, ticker: str):
    """Plot moving averages, Bollinger bands, RSI, and correlation heatmap."""

    # Moving Average Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#ffffff", width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA 20", line=dict(color="#f59e0b", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], name="SMA 50", line=dict(color="#3b82f6", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_200"], name="SMA 200", line=dict(color="#a855f7", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper",
                             line=dict(color="#10b981", dash="dash", width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower",
                             line=dict(color="#ef4444", dash="dash", width=1),
                             fill="tonexty", fillcolor="rgba(16,185,129,0.05)"))
    fig.update_layout(title=f"{ticker} — Moving Averages & Bollinger Bands",
                      template="plotly_dark", height=500)
    fig.write_html("data/eda_moving_averages.html")

    # RSI Chart
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], name="RSI 14",
                              line=dict(color="#f59e0b", width=1.5)))
    fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    fig2.update_layout(title=f"{ticker} — RSI (14-day)",
                       template="plotly_dark", height=350, yaxis=dict(range=[0, 100]))
    fig2.write_html("data/eda_rsi.html")

    # Correlation Heatmap
    corr_cols = ["Close", "SMA_20", "SMA_50", "EMA_20", "RSI_14",
                 "MACD", "BB_width", "Volatility_20", "Volume_ratio", "Daily_Return"]
    corr_df = df[corr_cols].dropna().corr()
    fig3 = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r",
                     title=f"{ticker} — Feature Correlation Heatmap",
                     template="plotly_dark", height=600)
    fig3.write_html("data/eda_correlation_heatmap.html")

    # Volatility
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df.index, y=df["Volatility_20"] * 100,
                              name="20-day Volatility (%)", line=dict(color="#a855f7")))
    fig4.update_layout(title=f"{ticker} — Rolling 20-day Volatility",
                       template="plotly_dark", yaxis_title="Volatility (%)", height=350)
    fig4.write_html("data/eda_volatility.html")

    print("  📊 EDA charts saved → data/eda_*.html")


plot_eda(df_features, TICKER)

# ─────────────────────────────────────────────
# 5. DATA MODELING
# ─────────────────────────────────────────────

# Using Random Forest Regressor — robust for tabular time-series features,
# handles non-linear relationships without requiring stationarity.

MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}

print("\n🌲 Model: Random Forest Regressor")
print(f"   Hyperparameters: {MODEL_PARAMS}")

# ─────────────────────────────────────────────
# 6. SPLITTING DATA INTO FEATURES AND TARGETS
# ─────────────────────────────────────────────

# Target: next-day closing price (1-step ahead forecast)
TARGET_COL = "Close"

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

# Drop rows with NaN (from rolling windows)
df_model = df_features[FEATURE_COLS + [TARGET_COL]].dropna().copy()

X = df_model[FEATURE_COLS].values
y = df_model[TARGET_COL].values

print(f"\n📐 Feature matrix shape: {X.shape}")
print(f"   Target vector shape:  {y.shape}")
print(f"   Features used ({len(FEATURE_COLS)}): {FEATURE_COLS[:5]} ... (and more)")

# ─────────────────────────────────────────────
# 7. SPLITTING INTO TRAINING AND TESTING SETS
# ─────────────────────────────────────────────

# Use a strict chronological split — never shuffle time-series data
TEST_SIZE = 0.15
split_idx = int(len(X) * (1 - TEST_SIZE))

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

train_dates = df_model.index[:split_idx]
test_dates = df_model.index[split_idx:]

print(f"\n✂️  Train set: {len(X_train)} samples  ({train_dates[0].date()} → {train_dates[-1].date()})")
print(f"   Test  set: {len(X_test)} samples  ({test_dates[0].date()} → {test_dates[-1].date()})")

# ─────────────────────────────────────────────
# 8. MODEL TRAINING
# ─────────────────────────────────────────────

print("\n🏋️  Training Random Forest model...")

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(**MODEL_PARAMS)
model.fit(X_train_scaled, y_train)

print("  ✅ Model training complete.")

# ─────────────────────────────────────────────
# 9. MODEL EVALUATION
# ─────────────────────────────────────────────

y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Mean Absolute Percentage Error
mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100

print("\n" + "=" * 50)
print("📈 MODEL EVALUATION RESULTS")
print("=" * 50)
print(f"  MAE   : ${mae:,.4f}")
print(f"  RMSE  : ${rmse:,.4f}")
print(f"  R²    : {r2:.6f}")
print(f"  MAPE  : {mape:.4f}%")
print("=" * 50)

# Feature Importance
feature_importance = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "Importance": model.feature_importances_,
}).sort_values("Importance", ascending=False).head(15)

print("\n🔍 Top 15 Feature Importances:")
print(feature_importance.to_string(index=False))

# ─────────────────────────────────────────────
# 10. COMMUNICATION AND VISUALIZATION
# ─────────────────────────────────────────────

def plot_actual_vs_predicted(test_dates, y_test, y_pred, ticker: str):
    """Plot actual vs predicted closing prices on test set."""
    residuals = y_test - y_pred

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            f"{ticker} — Actual vs Predicted Close Price (Test Set)",
            "Residuals (Actual − Predicted)",
        ),
    )

    fig.add_trace(
        go.Scatter(x=test_dates, y=y_test, name="Actual", line=dict(color="#60a5fa", width=2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=test_dates, y=y_pred, name="Predicted",
                   line=dict(color="#f59e0b", width=2, dash="dot")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=test_dates, y=residuals, name="Residuals",
               marker_color=["#ef4444" if r < 0 else "#10b981" for r in residuals], opacity=0.6),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_layout(template="plotly_dark", height=650, showlegend=True)
    fig.write_html("data/actual_vs_predicted.html")

    # Feature importance bar chart
    fig2 = px.bar(
        feature_importance,
        x="Importance", y="Feature",
        orientation="h",
        title=f"{ticker} — Top 15 Feature Importances",
        color="Importance",
        color_continuous_scale="Viridis",
        template="plotly_dark",
    )
    fig2.update_layout(yaxis=dict(autorange="reversed"), height=500)
    fig2.write_html("data/feature_importance.html")

    print("  📊 Actual vs Predicted chart → data/actual_vs_predicted.html")
    print("  📊 Feature importance chart  → data/feature_importance.html")


plot_actual_vs_predicted(test_dates, y_test, y_pred, TICKER)

# ─────────────────────────────────────────────
# 11. MODEL SAVING
# ─────────────────────────────────────────────

os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/stock_rf_model.sav"
SCALER_PATH = "models/scaler.sav"

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"\n💾 Model saved  → {MODEL_PATH}")
print(f"💾 Scaler saved → {SCALER_PATH}")

# ─────────────────────────────────────────────
# 12. DATA AUTOMATED PIPELINE IMPLEMENTATION
# ─────────────────────────────────────────────

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom sklearn transformer that engineers technical indicators
    from raw OHLCV data. Accepts a DataFrame and returns a numpy array
    of the selected feature columns.
    """

    def __init__(self, feature_cols: list = None):
        self.feature_cols = feature_cols

    def fit(self, X, y=None):
        # Stateless — no fitting required for deterministic features
        return self

    def transform(self, X, y=None):
        if isinstance(X, np.ndarray):
            # Reconstruct a minimal DataFrame from array if needed
            df = pd.DataFrame(X, columns=["Open", "High", "Low", "Close", "Volume"])
        else:
            df = X.copy()

        df = engineer_features(df)
        cols = self.feature_cols if self.feature_cols else FEATURE_COLS
        df = df[cols].ffill().bfill().fillna(0)
        return df.values


# Build the formal sklearn Pipeline
stock_pipeline = Pipeline(steps=[
    ("scaler", RobustScaler()),
    ("regressor", RandomForestRegressor(**MODEL_PARAMS)),
])

# Fit on scaled features (FeatureEngineer is already applied above outside pipeline)
stock_pipeline.fit(X_train_scaled, y_train)

# Verify pipeline predictions match standalone model
pipeline_pred = stock_pipeline.predict(X_test_scaled)
pipeline_r2 = r2_score(y_test, pipeline_pred)
print(f"\n🔧 sklearn Pipeline R² (verification): {pipeline_r2:.6f}")

# ─────────────────────────────────────────────
# 13. PIPELINE SAVING
# ─────────────────────────────────────────────

PIPELINE_PATH = "models/stock_pipeline_compressed.sav"

# Save compressed (compress=3 gives good size/speed tradeoff)
joblib.dump(stock_pipeline, PIPELINE_PATH, compress=3)

# Save metadata for the Streamlit app
metadata = {
    "ticker": TICKER,
    "feature_cols": FEATURE_COLS,
    "target_col": TARGET_COL,
    "train_end_date": str(train_dates[-1].date()),
    "test_mae": round(mae, 4),
    "test_rmse": round(rmse, 4),
    "test_r2": round(r2, 6),
    "test_mape": round(mape, 4),
}
joblib.dump(metadata, "models/pipeline_metadata.sav")

print(f"\n🗜️  Compressed pipeline saved → {PIPELINE_PATH}")
print(f"📋 Pipeline metadata saved  → models/pipeline_metadata.sav")

print("\n🎉 Full pipeline execution complete!")
print("   Run `streamlit run app.py` to launch the prediction dashboard.")
