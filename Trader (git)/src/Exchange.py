import pandas as pd
import os
import time
import requests

class Exchange:
    def __init__(self, source, symbol="BTCUSDT", candle_interval="5m", backtest_file_path=None):
        self.source = source.lower()
        self.symbol = symbol.upper()
        self.candle_interval = candle_interval
        self.index = 0

        if self.source == "backtest":
            if not backtest_file_path:
                raise ValueError("Backtest file path must be provided for backtest mode.")
            self.df = pd.read_csv(backtest_file_path)
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
            self.df = self.df.sort_values(by="timestamp").reset_index(drop=True)
        elif self.source == "binance":
            self.api_url = "https://api.binance.com/api/v3/klines"
        else:
            raise ValueError("Unsupported exchange source. Use 'binance' or 'backtest'.")

    def next_candle(self):
        if self.source == "backtest":
            if self.index >= len(self.df):
                return None  # End of data
            candle = self.df.iloc[self.index].to_dict()
            self.index += 1
            return candle

        elif self.source == "binance":
            params = {
                "symbol": self.symbol,
                "interval": self.candle_interval,
                "limit": 1
            }
            response = requests.get(self.api_url, params=params)
            if response.status_code != 200:
                raise Exception("Binance API request failed: {}".format(response.text))

            kline = response.json()[0]
            return {
                "timestamp": pd.to_datetime(kline[0], unit='ms'),
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5])
            }
