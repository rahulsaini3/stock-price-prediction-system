"""
StockSense AI — Standalone Python Script
Bina Streamlit ke terminal mein chalane ke liye
Usage: python stock_model.py --ticker AAPL --period 5y --forecast 30
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Try TensorFlow
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
    print("✅ TensorFlow available — LSTM enabled")
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not found — only Linear Regression will run")
    print("    Install with: pip install tensorflow")

plt.style.use('dark_background')
COLORS = {
    'bg':     '#0a0e1a',
    'card':   '#111827',
    'accent': '#00d4ff',
    'purple': '#7b2fff',
    'green':  '#00e676',
    'red':    '#ff5252',
    'gold':   '#ffd700',
    'text':   '#8899cc',
}


# ─── Data ─────────────────────────────────────────────────────────────────────

def fetch_data(ticker: str, period: str = '5y') -> pd.DataFrame:
    print(f"\n📥 Fetching {ticker} data ({period})...")
    stock = yf.Ticker(ticker)
    df    = stock.history(period=period)
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    df = df.dropna()
    print(f"   Rows: {len(df)} | From: {df.index[0].date()} to {df.index[-1].date()}")
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for w in [20, 50, 100, 200]:
        df[f'MA{w}'] = df['Close'].rolling(w).mean()
    # RSI
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    # Bollinger
    ma  = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['BB_Up'] = ma + 2*std
    df['BB_Dn'] = ma - 2*std
    return df


# ─── Models ───────────────────────────────────────────────────────────────────

def linear_regression_predict(close_prices, split_idx):
    X = np.arange(len(close_prices)).reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(X[:split_idx], close_prices[:split_idx])
    pred  = lr.predict(X[split_idx:])
    rmse  = np.sqrt(mean_squared_error(close_prices[split_idx:], pred))
    print(f"\n📐 Linear Regression RMSE: ${rmse:.2f}")
    return pred, rmse


def prepare_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)


def train_lstm(scaled_train, scaled_test, scaler, seq_len=60, epochs=30, forecast_days=30):
    if not TF_AVAILABLE:
        return None, None, None

    print(f"\n🧠 Training LSTM (seq_len={seq_len}, epochs={epochs})...")

    X_tr, y_tr = prepare_sequences(scaled_train, seq_len)
    X_ts, y_ts = prepare_sequences(scaled_test,  seq_len)
    X_tr = X_tr.reshape(-1, seq_len, 1)
    X_ts = X_ts.reshape(-1, seq_len, 1)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    es = EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(X_tr, y_tr, epochs=epochs, batch_size=32,
                        validation_split=0.1, callbacks=[es], verbose=1)

    pred_scaled = model.predict(X_ts, verbose=0)
    lstm_pred   = scaler.inverse_transform(pred_scaled)
    actual      = scaler.inverse_transform(y_ts.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(actual, lstm_pred))
    print(f"🎯 LSTM RMSE: ${rmse:.2f}")

    # Future forecast
    all_scaled = np.concatenate([scaled_train, scaled_test])
    last_seq   = all_scaled[-seq_len:].reshape(1, seq_len, 1)
    future = []
    cur    = last_seq.copy()
    for _ in range(forecast_days):
        p = model.predict(cur, verbose=0)[0][0]
        future.append(p)
        cur = np.append(cur[:, 1:, :], [[[p]]], axis=1)
    forecast = scaler.inverse_transform(np.array(future).reshape(-1, 1)).flatten()

    return lstm_pred, rmse, forecast


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_technical(df, ticker):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
                              facecolor=COLORS['bg'],
                              gridspec_kw={'height_ratios': [3, 1, 1]})
    fig.suptitle(f'{ticker} — Technical Analysis', color=COLORS['accent'],
                 fontsize=16, fontweight='bold', y=0.98)

    ax1, ax2, ax3 = axes
    for ax in axes:
        ax.set_facecolor(COLORS['card'])
        ax.tick_params(colors=COLORS['text'])
        ax.grid(alpha=0.15, color=COLORS['text'])

    ax1.fill_between(df.index, df['BB_Up'], df['BB_Dn'], alpha=0.07, color=COLORS['purple'])
    ax1.plot(df.index, df['BB_Up'], color=COLORS['purple'], alpha=0.4, lw=1)
    ax1.plot(df.index, df['BB_Dn'], color=COLORS['purple'], alpha=0.4, lw=1)
    ax1.plot(df.index, df['Close'],  color=COLORS['accent'], lw=1.5, label='Close', zorder=5)
    ax1.plot(df.index, df['MA20'],   color='#00d4ff', lw=1.2, alpha=0.7, label='MA20')
    ax1.plot(df.index, df['MA50'],   color=COLORS['gold'],  lw=1.2, alpha=0.7, label='MA50')
    ax1.plot(df.index, df['MA100'],  color=COLORS['red'],   lw=1.2, alpha=0.7, label='MA100')
    ax1.plot(df.index, df['MA200'],  color='#ff00ff', lw=1.2, alpha=0.7, label='MA200')
    ax1.legend(facecolor=COLORS['bg'], labelcolor=COLORS['text'], fontsize=8)
    ax1.set_ylabel('Price ($)', color=COLORS['text'])

    colors_v = [COLORS['green'] if c >= o else COLORS['red']
                for c, o in zip(df['Close'], df['Open'])]
    ax2.bar(df.index, df['Volume'], color=colors_v, alpha=0.7, width=0.8)
    ax2.set_ylabel('Volume', color=COLORS['text'])

    ax3.plot(df.index, df['RSI'], color=COLORS['gold'], lw=1.5)
    ax3.axhline(70, color=COLORS['red'],   ls='--', alpha=0.7, label='Overbought')
    ax3.axhline(30, color=COLORS['green'], ls='--', alpha=0.7, label='Oversold')
    ax3.fill_between(df.index, df['RSI'], 70, where=df['RSI']>=70,
                     alpha=0.2, color=COLORS['red'])
    ax3.fill_between(df.index, df['RSI'], 30, where=df['RSI']<=30,
                     alpha=0.2, color=COLORS['green'])
    ax3.set_ylim(0, 100)
    ax3.set_ylabel('RSI', color=COLORS['text'])
    ax3.legend(facecolor=COLORS['bg'], labelcolor=COLORS['text'], fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{ticker}_technical.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'])
    print(f"💾 Saved: {ticker}_technical.png")
    plt.show()


def plot_predictions(df, split_idx, lr_pred, lstm_pred=None,
                     lstm_forecast=None, forecast_days=30,
                     lr_rmse=None, lstm_rmse=None, ticker='STOCK'):
    fig, ax = plt.subplots(figsize=(16, 7), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['card'])
    ax.tick_params(colors=COLORS['text'])
    ax.grid(alpha=0.15, color=COLORS['text'])

    close = df['Close'].values
    dates = df.index

    ax.plot(dates[:split_idx], close[:split_idx],
            color=COLORS['text'], lw=1.2, alpha=0.6, label='Training Data')
    ax.plot(dates[split_idx:], close[split_idx:],
            color=COLORS['green'], lw=2, label='Actual Price')
    ax.plot(dates[split_idx:], lr_pred.flatten(),
            color=COLORS['gold'], lw=2, ls='--',
            label=f'Linear Regression (RMSE: ${lr_rmse:.2f})')

    if lstm_pred is not None:
        seq_len = len(dates[split_idx:]) - len(lstm_pred)
        lstm_dates = dates[split_idx + seq_len:]
        ax.plot(lstm_dates, lstm_pred.flatten(),
                color=COLORS['red'], lw=2.5,
                label=f'LSTM (RMSE: ${lstm_rmse:.2f})')

    if lstm_forecast is not None:
        last_date    = dates[-1]
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1),
                                     periods=forecast_days, freq='B')
        ax.plot(future_dates, lstm_forecast,
                color=COLORS['purple'], lw=2.5, ls=':',
                marker='o', ms=3, label=f'{forecast_days}-Day Forecast')
        ax.fill_between(future_dates, lstm_forecast*0.97, lstm_forecast*1.03,
                        alpha=0.15, color=COLORS['purple'])
        ax.axvline(dates[-1], color=COLORS['purple'], ls='--', alpha=0.5)
        ax.text(dates[-1], ax.get_ylim()[1]*0.98, '  Forecast →',
                color=COLORS['purple'], fontsize=9)

    ax.set_title(f'{ticker} — Actual vs Predicted', color=COLORS['accent'],
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', color=COLORS['text'])
    ax.set_ylabel('Price ($)', color=COLORS['text'])
    ax.legend(facecolor=COLORS['bg'], labelcolor=COLORS['text'])

    plt.tight_layout()
    plt.savefig(f'{ticker}_prediction.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'])
    print(f"💾 Saved: {ticker}_prediction.png")
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='StockSense AI Predictor')
    parser.add_argument('--ticker',   default='AAPL',  help='Stock ticker (e.g. AAPL, RELIANCE.NS)')
    parser.add_argument('--period',   default='5y',    help='Data period (2y, 5y, 10y)')
    parser.add_argument('--seq_len',  default=60,  type=int, help='LSTM sequence length')
    parser.add_argument('--epochs',   default=30,  type=int, help='Training epochs')
    parser.add_argument('--forecast', default=30,  type=int, help='Forecast days')
    args = parser.parse_args()

    print("="*55)
    print("       📈 StockSense AI — Prediction Engine")
    print("="*55)

    df    = fetch_data(args.ticker, args.period)
    df    = add_indicators(df)

    close  = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close)

    split = int(len(scaled) * 0.8)

    # Linear Regression
    lr_pred, lr_rmse = linear_regression_predict(close, split)

    # LSTM
    lstm_pred, lstm_rmse, lstm_forecast = None, None, None
    if TF_AVAILABLE:
        lstm_pred, lstm_rmse, lstm_forecast = train_lstm(
            scaled[:split], scaled[split:], scaler,
            seq_len=args.seq_len, epochs=args.epochs,
            forecast_days=args.forecast
        )

    # Summary
    print("\n" + "="*55)
    print("📊 RESULTS SUMMARY")
    print("="*55)
    print(f"  Stock          : {args.ticker}")
    print(f"  Current Price  : ${df['Close'].iloc[-1]:.2f}")
    print(f"  Data Points    : {len(df)}")
    print(f"  Train/Test     : {split} / {len(df)-split}")
    print(f"  LR RMSE        : ${lr_rmse:.2f}")
    if lstm_rmse:
        print(f"  LSTM RMSE      : ${lstm_rmse:.2f}")
        improvement = ((lr_rmse - lstm_rmse) / lr_rmse) * 100
        print(f"  Improvement    : {improvement:.1f}%")
    if lstm_forecast is not None:
        print(f"  {args.forecast}-Day Forecast  : ${lstm_forecast[-1]:.2f}")
    print("="*55)

    # Plots
    plot_technical(df, args.ticker)
    plot_predictions(df, split, lr_pred, lstm_pred, lstm_forecast,
                     args.forecast, lr_rmse, lstm_rmse, args.ticker)


if __name__ == '__main__':
    main()
