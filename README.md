# 📈 StockSense AI — Stock Price Predictor

> AI-powered stock price prediction using LSTM + Linear Regression | Built with Python, Streamlit & TensorFlow

---

## 🗂️ Project Structure

```
stock_predictor/
├── app.py              ← Streamlit Web App (Main UI)
├── stock_model.py      ← Standalone Python Script (Terminal use)
├── requirements.txt    ← All dependencies
└── README.md
```

---

## ⚙️ Setup (Pehli baar install karna)

### Step 1: Python Environment banayein (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 2: Dependencies install karein
```bash
pip install -r requirements.txt
```

> ⚠️ TensorFlow large package hai (~500MB). Agar sirf basic prediction chahiye:
> ```bash
> pip install streamlit yfinance pandas numpy scikit-learn plotly
> ```

---

## 🚀 Run Kaise Karein

### Option 1: Streamlit Web App (Recommended)
```bash
streamlit run app.py
```
Browser mein automatically `http://localhost:8501` open hoga.

### Option 2: Terminal Script
```bash
# Basic (AAPL, 5 years data, 30-day forecast):
python stock_model.py

# Custom ticker:
python stock_model.py --ticker RELIANCE.NS --period 5y --forecast 30

# All options:
python stock_model.py --ticker TCS.NS --period 10y --seq_len 60 --epochs 50 --forecast 60
```

---

## 📊 Features

| Feature | Description |
|---------|-------------|
| 📥 Data Fetch | yfinance se automatic historical data |
| 🕯️ Candlestick Chart | OHLC with Bollinger Bands |
| 📉 Moving Averages | MA20, MA50, MA100, MA200 |
| 📈 RSI Indicator | Overbought/Oversold zones |
| 🤖 Linear Regression | Baseline prediction model |
| 🧠 LSTM Neural Network | Advanced deep learning model |
| 🔮 Future Forecast | N-day price forecast with confidence band |
| 📊 RMSE Comparison | Model accuracy comparison |
| 💹 Returns Analysis | Daily returns distribution + volatility |

---

## 🎯 Popular Ticker Symbols

| Company | Ticker |
|---------|--------|
| Apple | `AAPL` |
| Google | `GOOGL` |
| Tesla | `TSLA` |
| Microsoft | `MSFT` |
| Reliance | `RELIANCE.NS` |
| TCS | `TCS.NS` |
| Infosys | `INFY.NS` |
| HDFC Bank | `HDFCBANK.NS` |
| Wipro | `WIPRO.NS` |

---

## 🧠 Model Architecture (LSTM)

```
Input: (batch, seq_len=60, features=1)
    ↓
LSTM(64 units, return_sequences=True)
    ↓
Dropout(0.2)
    ↓
LSTM(64 units)
    ↓
Dropout(0.2)
    ↓
Dense(32, relu)
    ↓
Dense(1)  ← Predicted price
```

**Training:** 80% data | **Testing:** 20% data
**Loss:** Mean Squared Error | **Optimizer:** Adam

---

## 📐 Evaluation Metrics

- **RMSE (Root Mean Squared Error):** Jitna kam, utna better
  ```
  RMSE = √(Σ(actual - predicted)² / n)
  ```

---

## ⚠️ Disclaimer
Yeh project sirf **educational aur academic purpose** ke liye hai.
Real investment decisions ke liye professional financial advisor se consult karein.
Stock market predictions 100% accurate nahi hoti.

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **yfinance** — Data collection
- **Pandas / NumPy** — Data processing
- **Scikit-Learn** — Linear Regression + MinMaxScaler
- **TensorFlow / Keras** — LSTM model
- **Plotly** — Interactive charts
- **Streamlit** — Web UI
