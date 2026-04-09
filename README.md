# Stock-Market-Predictor-Pro
"A Real-time Stock Market Prediction system using Long Short-Term Memory (LSTM) Deep Learning model. Features live data extraction via Yahoo Finance API and an interactive web dashboard built with Streamlit."

# 📈 Real-Time Stock Market Prediction Pro

An end-to-end Deep Learning project designed to predict stock market trends using live data. This project utilizes the power of **LSTM (Long Short-Term Memory)** networks to analyze historical price patterns and forecast future closing prices.

## 🚀 Project Overview
This project follows a professional data science pipeline to provide accurate stock price visualizations and predictions. It bypasses static datasets by fetching real-time data directly from financial markets.

## 🛠 Tech Stack
* **Language:** Python 3.12
* **Libraries:** TensorFlow, Keras, Scikit-Learn, Pandas, Numpy
* **Data Source:** Yahoo Finance API (`yfinance`)
* **Visualization:** Matplotlib, Streamlit
* **Deployment:** Streamlit Cloud

## 📋 Project Workflow
I have structured this project into a 13-step professional pipeline:
1. **Importing Dependencies:** Setting up the environment.
2. **Data Cleaning:** Handling missing values and indexing.
3. **Data Visualization:** Plotting historical price trends.
4. **Exploratory Data Analysis (EDA):** Calculating Moving Averages (MA100, MA200).
5. **Data Modeling:** Feature scaling using MinMaxScaler.
6. **Feature Engineering:** Creating 60-day time-step sequences.
7. **Data Splitting:** Time-series based Train-Test split (80/20).
8. **Model Training:** Building a Stacked LSTM architecture with Dropout layers.
9. **Model Evaluation:** Calculating RMSE and R-squared scores.
10. **Results Communication:** Visualizing Actual vs. Predicted prices.
11. **Model Saving:** Exporting as `.sav` file using Joblib.
12. **Automated Pipeline:** Creating a function for real-time inference.
13. **Pipeline Saving:** Compressed storage for deployment.

## 📊 Model Architecture

The model uses a multi-layered LSTM approach to capture long-term dependencies in stock price movements, optimized with the **Adam** optimizer and **Mean Squared Error (MSE)** loss function.

## 🖥 How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/stock-market-prediction.git](https://github.com/yourusername/stock-market-prediction.git)
