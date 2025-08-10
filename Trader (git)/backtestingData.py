# backtester.py
import csv
import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from requests.exceptions import RequestException, ConnectionError
from urllib3.exceptions import ProtocolError

INTERVALS = ["5m"]
CANDLE_LIMIT = 48  # Not used anymore directly
DAYS_LIMIT = 2
START_DATE = "2025-06-01 00:00:00"  # Set your desired start date here
OUTPUT_DIR = "../backtest_data_20250602-01"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_spot_tickers(currency="USDT"):
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

def fetch_binance_candles_from_start(ticker, interval, start_date, days_limit, retries=5, delay=3):
    start_time = int(pd.to_datetime(start_date).timestamp() * 1000)
    end_time = int((pd.to_datetime(start_date) + timedelta(days=days_limit)).timestamp() * 1000)
    all_candles = []

    while start_time < end_time:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": ticker,
            "interval": interval,
            "startTime": start_time,
            "limit": 1000
        }

        for attempt in range(retries):
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    batch = response.json()
                    if not batch:
                        return all_candles
                    all_candles.extend(batch)
                    start_time = batch[-1][0] + 1  # move start_time to just after last returned candle
                    break
                elif response.status_code == 429:
                    wait_time = delay + random.uniform(1, 3)
                    print(f"[RATE LIMIT] Too many requests. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                elif response.status_code == 418:
                    print("[BANNED] IP temporarily banned by Binance. Exiting.")
                    return all_candles
                else:
                    print(f"[HTTP ERROR {response.status_code}] {response.text}")
                    return all_candles
            except (RequestException, ConnectionError, ProtocolError) as e:
                print(f"[RETRY {attempt + 1}] Network error: {e}. Waiting {delay} seconds before retry...")
                time.sleep(delay)
                delay *= 2

    return all_candles

def save_candles_to_csv(ticker, interval, candles):
    if not candles:
        return

    filename = os.path.join(OUTPUT_DIR, f"{ticker}_{interval}.csv")
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for c in candles:
            writer.writerow([
                datetime.fromtimestamp(c[0] / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                c[1], c[2], c[3], c[4], c[5]
            ])

def main():
    tickers = get_spot_tickers("USDT")
    print(f"Fetched {len(tickers)} tickers")

    for ticker in tickers:
        print(f"Fetching {ticker}...")
        for interval in INTERVALS:
            candles = fetch_binance_candles_from_start(ticker, interval, START_DATE, DAYS_LIMIT)
            save_candles_to_csv(ticker, interval, candles)
            time.sleep(0.1)

if __name__ == "__main__":
    main()
