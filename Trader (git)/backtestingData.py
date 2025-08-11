import os
import time
import requests
import pandas as pd
from datetime import timedelta
from requests.exceptions import RequestException, ConnectionError
from urllib3.exceptions import ProtocolError

# ========================
# FIXED SETTINGS
# ========================
START_DATE = "2025-06-20 00:00:00"   # start datetime
DAYS_LIMIT = 5                       # number of days to fetch
INTERVAL = "5m"                       # candle interval
OUTPUT_DIR = "../backtest_data_20250620-5days"  # where to save CSVs
TARGET_TICKERS = ["BTCUSDT", "SOLUSDT", "ETHUSDT"]  # [] = all USDT spot
# ========================


def get_spot_tickers(quote="USDT"):
    url = "https://api.binance.com/api/v3/exchangeInfo"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()["symbols"]
    return [
        s["symbol"]
        for s in data
        if s.get("isSpotTradingAllowed")
        and s.get("status") == "TRADING"
        and s.get("quoteAsset") == quote
    ]


def fetch_klines_since(ticker, interval, start_date, days, retries=5, delay=3):
    start = int(pd.to_datetime(start_date).timestamp() * 1000)
    end = int(
        (pd.to_datetime(start_date) + timedelta(days=days)).timestamp() * 1000
    )
    out = []
    while start < end:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": ticker,
            "interval": interval,
            "startTime": start,
            "limit": 1000,
        }
        for attempt in range(retries):
            try:
                res = requests.get(url, params=params, timeout=15)
                if res.status_code == 429:  # rate limit
                    time.sleep(60)
                    continue
                res.raise_for_status()
                batch = res.json()
                if not batch:
                    return out
                out.extend(batch)
                start = batch[-1][6] + 1  # move to next candle
                time.sleep(0.2)
                break
            except (RequestException, ConnectionError, ProtocolError):
                if attempt == retries - 1:
                    raise
                time.sleep(delay * (attempt + 1))
    return out


def save_csv(ticker, interval, candles, out_dir):
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(candles, columns=cols)
    df = df[["open_time", "open", "high", "low", "close", "volume", "close_time", "num_trades"]]
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    fp = os.path.join(out_dir, f"{ticker}_{interval}.csv")
    df.to_csv(fp, index=False)
    return fp


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if TARGET_TICKERS:
        tickers = TARGET_TICKERS
    else:
        tickers = get_spot_tickers("USDT")

    print(f"Fetching {len(tickers)} tickers → {OUTPUT_DIR}")
    for i, t in enumerate(tickers, 1):
        try:
            kl = fetch_klines_since(t, INTERVAL, START_DATE, DAYS_LIMIT)
            if kl:
                path = save_csv(t, INTERVAL, kl, OUTPUT_DIR)
                print(f"[{i}/{len(tickers)}] {t}: {len(kl)} candles → {path}")
            else:
                print(f"[{i}/{len(tickers)}] {t}: no data")
        except Exception as e:
            print(f"[{i}/{len(tickers)}] {t}: ERROR {e}")


if __name__ == "__main__":
    main()