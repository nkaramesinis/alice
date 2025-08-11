# generate_ml_dataset.py

import os
import pandas as pd
import ta

# Configuration
DATA_DIR = "../../backtest_data"
OUTPUT_FILE = "../ml_dataset.csv"
PROFIT_TARGET_PCT = 0.03
STOP_LOSS_PCT = 0.01
MAX_HOLDING_CANDLES = 15


def simulate_trade(entry_idx, df):
    entry_price = df.loc[entry_idx, "close"]
    for i in range(entry_idx + 1, min(entry_idx + MAX_HOLDING_CANDLES + 1, len(df))):
        high = df.loc[i, "high"]
        low = df.loc[i, "low"]

        if high >= entry_price * (1 + PROFIT_TARGET_PCT):
            return 1  # Profit
        if low <= entry_price * (1 - STOP_LOSS_PCT):
            return 0  # Loss
    return 0  # No profit achieved


def process_file(file_path, ticker):
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["ticker"] = ticker

    # Convert types
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

    # Indicators
    df["ema_9"] = ta.trend.ema_indicator(df["close"], window=9)
    df["ema_21"] = ta.trend.ema_indicator(df["close"], window=21)
    df["ema_50"] = ta.trend.ema_indicator(df["close"], window=50)
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["avg_volume"] = df["volume"].rolling(window=14).mean()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_width"] = df["bb_high"] - df["bb_low"]

    # Drop NaNs
    df = df.dropna().reset_index(drop=True)

    # Label
    rows = []
    for i in range(50, len(df) - MAX_HOLDING_CANDLES):
        row = df.iloc[i]
        label = simulate_trade(i, df)
        rows.append({
            "ticker": row["ticker"],
            "timestamp": row["timestamp"],
            "close": row["close"],
            "ema_gap": (row["ema_9"] - row["ema_21"]) / row["ema_21"],
            "rsi": row["rsi"],
            "volume_ratio": row["volume"] / row["avg_volume"] if row["avg_volume"] > 0 else 1,
            "bb_width": row["bb_width"],
            "above_ema_50": row["close"] > row["ema_50"],
            "label": label
        })
    return pd.DataFrame(rows)


def main():
    dataset = []
    for file in os.listdir(DATA_DIR):
        if file.endswith("5m.csv"):
            ticker = file.replace("_5m.csv", "")
            file_path = os.path.join(DATA_DIR, file)
            df = process_file(file_path, ticker)
            dataset.append(df)

    all_data = pd.concat(dataset).reset_index(drop=True)
    all_data.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset saved to {OUTPUT_FILE} with {len(all_data)} rows.")


if __name__ == "__main__":
    main()
