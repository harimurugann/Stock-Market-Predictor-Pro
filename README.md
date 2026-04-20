<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:00ff88,100:00d4ff&height=200&section=header&text=StockSight%20AI&fontSize=60&fontColor=ffffff&fontAlignY=38&desc=End-to-End%20ML%20Stock%20Prediction%20Pipeline&descAlignY=60&descSize=16&animation=fadeIn" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.4.2-orange?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.35.0-red?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/yfinance-0.2.40-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

<br/>

## 🧠 Overview

**StockSight AI** is a professional-grade, end-to-end stock market prediction pipeline built in Python. It fetches **live market data** via `yfinance`, engineers **33 technical indicators**, trains a **Random Forest Regressor** inside a formal `sklearn Pipeline`, and serves predictions through a sleek **Streamlit dashboard** with a retro-futuristic theme.

---

## 📁 Folder Structure

```
stocksight-ai/
│
├── app.py                          # Streamlit prediction dashboard
├── stock_prediction_pipeline.py    # Full training pipeline (headings 1–13)
├── requirements.txt                # Pinned dependencies
├── README.md
├── .gitignore
│
├── data/                           # Auto-generated HTML charts
│   ├── candlestick_chart.html
│   ├── eda_moving_averages.html
│   ├── eda_rsi.html
│   ├── eda_volatility.html
│   ├── eda_correlation_heatmap.html
│   ├── actual_vs_predicted.html
│   └── feature_importance.html
│
├── models/                         # Saved artifacts (git-ignored)
│   ├── stock_rf_model.sav
│   ├── scaler.sav
│   ├── stock_pipeline_compressed.sav
│   └── pipeline_metadata.sav
│
└── notebooks/                      # Optional: Jupyter EDA notebooks
    └── eda_exploration.ipynb
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/stocksight-ai.git
cd stocksight-ai
pip install -r requirements.txt
```

### 2. Train the Pipeline

```bash
python stock_prediction_pipeline.py
```

This will:
- Fetch live AAPL data via `yfinance`
- Engineer 33 technical features
- Train a Random Forest model
- Save `.sav` artifacts to `models/`
- Export 7 interactive HTML charts to `data/`

### 3. Launch Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🏗️ Pipeline Architecture

```
yfinance (Live Data)
        │
        ▼
┌───────────────────────┐
│  Data Cleaning        │  → Drop nulls, IQR outlier removal
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  Feature Engineering  │  → 33 technical indicators
│  (SMA, EMA, RSI,      │
│   MACD, BB, Lags...)  │
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  TimeSeriesSplit      │  → Chronological 85/15 split
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  sklearn Pipeline     │  → RobustScaler → RandomForest
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  Evaluation           │  → MAE, RMSE, R², MAPE
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  Compressed .sav      │  → joblib.dump(compress=3)
└───────────────────────┘
        │
        ▼
   Streamlit App 🚀
```

---

## 📊 Features Engineered (33 Total)

| Category | Features |
|---|---|
| Price | Open, High, Low, Volume |
| SMA | 5, 10, 20, 50-day |
| EMA | 5, 10, 20-day |
| Bollinger Bands | Upper, Lower, Width |
| Momentum | 5-day, 10-day |
| RSI | 14-day |
| MACD | MACD, Signal, Histogram |
| Volatility | 5-day, 20-day rolling std |
| Lags | Close lag 1,2,3,5 · Return lag 1,2,3 |
| Volume | Volume ratio (vs 10-day avg) |
| Calendar | Day of week, Month, Quarter |

---

## 📈 Model Details

| Parameter | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Estimators | 300 trees |
| Max Depth | 10 |
| Scaler | RobustScaler |
| Split Strategy | Chronological (no shuffle) |
| Test Set | Last 15% of data |

---

## 🖥️ Dashboard Features

- 🕯️ **Interactive Candlestick** chart with toggleable overlays
- 📉 **RSI & MACD** indicator panels
- 🤖 **Next-day price prediction** with directional signal
- 📋 **Live metrics** (open, close, volume, range)
- 🏆 **Model performance** metrics displayed in-app

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `yfinance` | Live market data |
| `scikit-learn` | ML pipeline & modeling |
| `joblib` | Model serialization |
| `plotly` | Interactive charts |
| `streamlit` | Web dashboard |
| `pandas` / `numpy` | Data processing |

---

## 📄 License

MIT License — feel free to fork, modify, and build on this project.

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:00d4ff,100:00ff88&height=100&section=footer&animation=fadeIn" />
</p>
