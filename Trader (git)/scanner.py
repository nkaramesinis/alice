# scanner.py
import time
import pandas as pd
import ta
import csv
import requests
from requests.exceptions import RequestException

# Configuration
DATA_SOURCE = "binance"  # Options: 'binance', 'file'
CSV_FILE = "top_50_binance_tickers.csv"  # Path to tickers list
INTERVAL = "5m"
CANDLE_LIMIT = 48  # 4 hours of 5-minute candles
PRICE_CHANGE_THRESHOLD = 0.20  # 20%
VOLUME_INCREASE_THRESHOLD = 0.5  # 50%
VOLATILITY_THRESHOLD = 0.01  # Adjust volatility threshold as needed
VOLATILITY_CANDLE_LOOKBACK = 14  # Number of candles to calculate volatility over


def get_spot_tickers(currency):
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url)
    data = response.json()

    spot_tickers = [
        s["symbol"] for s in data["symbols"]
        if s["status"] == "TRADING"
        and s["quoteAsset"] == currency
        and s["isSpotTradingAllowed"]
    ]

    return spot_tickers


def load_tickers_from_csv(file_path):
    tickers = []
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            tickers.append(row["ticker"])
    return tickers


def fetch_binance_candles(ticker, interval, limit=None, retries=3, delay=5):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": ticker,
        "interval": interval
    }
    if limit is not None:
        params["limit"] = limit

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print(f"[RATE LIMIT] Too many requests. Waiting {delay} seconds...")
                time.sleep(delay)
            elif response.status_code == 418:
                print(f"[BANNED] IP has been banned temporarily by Binance.")
                break
            else:
                print(f"[ERROR {response.status_code}] {response.text}")
                break
        except RequestException as e:
            print(f"[REQUEST ERROR] {e}")
            time.sleep(delay)
    return []


def fetch_candle_data(ticker):
    candles = fetch_binance_candles(ticker, INTERVAL, CANDLE_LIMIT)
    if not candles:
        return None

    df = pd.DataFrame(candles, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])

    # Remove rows with missing values
    df = df.replace("null", None).dropna(subset=["high", "low", "close", "volume"])

    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)

    # Add RSI, MACD
    df["rsi"] = ta.momentum.RSIIndicator(close=df["close"]).rsi()
    macd = ta.trend.MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    # Add price range for volatility
    df["range"] = df["high"] - df["low"]
    df["volatility"] = df["range"].rolling(window=VOLATILITY_CANDLE_LOOKBACK).mean()

    return df


def is_bullish(df):
    if df is None or len(df) < 30:
        return False, 0, 0, 0, 0, 0

    start_price = df["close"].iloc[0]
    end_price = df["close"].iloc[-1]
    price_change = (end_price - start_price) / start_price

    midpoint = len(df) // 2
    avg_volume_first_half = df["volume"].iloc[:midpoint].mean()
    avg_volume_second_half = df["volume"].iloc[midpoint:].mean()
    volume_increase = (avg_volume_second_half - avg_volume_first_half) / avg_volume_first_half

    rsi_slope = df["rsi"].iloc[-1] - df["rsi"].iloc[-5]
    macd_value = df["macd"].iloc[-1]
    rsi_value = df["rsi"].iloc[-1]
    volatility_value = df["volatility"].iloc[-1] / df["close"].iloc[-1]  # Normalize

    return (
        price_change >= PRICE_CHANGE_THRESHOLD
        and volume_increase >= VOLUME_INCREASE_THRESHOLD
        and volatility_value >= VOLATILITY_THRESHOLD
    ), price_change, volume_increase, macd_value, rsi_value, volatility_value


def run_scanner():
    tickers = get_spot_tickers("USDT")
    matching = []
    top_3 = []
    for ticker in tickers:
        df = fetch_candle_data(ticker)
        if df is None:
            print(f"Skipping {ticker}, could not fetch data.")
            continue

        result, price_change, volume_increase, macd_val, rsi_val, vol_val = is_bullish(df)
        print(f"Analyzing: {ticker} | Price Change: {price_change:.2%} | Volume Increase: {volume_increase:.2%} | Volatility: {vol_val:.2%}")
        if result:
            print(f"Bullish!!!!: {ticker} | Price Change: {price_change:.2%} | Volume Increase: {volume_increase:.2%} | MACD: {macd_val:.4f} | RSI: {rsi_val:.2f} | Volatility: {vol_val:.2%}")
            matching.append((ticker, price_change, volume_increase, macd_val, rsi_val, vol_val))
            top_3 = matching[:3]

    print("\nTop candidates:")
    for t in top_3:
        print(f"{t[0]} | Price Change: {t[1]:.2%}, Volume Increase: {t[2]:.2%}, MACD: {t[3]:.4f}, RSI: {t[4]:.2f}, Volatility: {t[5]:.2%}")


def main():
    run_scanner()


if __name__ == "__main__":
    main()
