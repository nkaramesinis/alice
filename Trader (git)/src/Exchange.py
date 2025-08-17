# src/exchange.py
from __future__ import annotations
import os, glob, time
from typing import Dict, Optional, List

import pandas as pd
import requests

Candle = Dict[str, float | int]


class Exchange:
    """
    Uniform candle source for backtests and live:
      - backtest: reads {SYMBOL}_{INTERVAL}.csv with lowercase columns from data_dir
      - binance : polls latest *closed* kline (you can swap to websockets later)

    Always returns a dict:
      {"ts", "open", "high", "low", "close", "volume"}
    where ts is an int (ms since epoch), derived from CSV close_time in backtests.
    """

    def __init__(
        self,
        mode: str,
        interval: str,
        symbols: List[str],
        data_dir: Optional[str] = None,
        binance_base_url: str = "https://api.binance.com",
    ):
        self.mode = mode.lower().strip()
        self.interval = interval
        self.symbols = [s.upper() for s in symbols]
        self.data_dir = data_dir
        self.binance_base_url = binance_base_url.rstrip("/")

        self._frames: Dict[str, pd.DataFrame] = {}
        self._idx: Dict[str, int] = {}

        if self.mode == "backtest":
            if not self.data_dir:
                raise ValueError("data_dir is required in backtest mode")
            self._load_backtest_frames()
        elif self.mode == "binance":
            pass
        else:
            raise ValueError("mode must be 'backtest' or 'binance'")

    # ------------------------------
    # Backtest helpers
    # ------------------------------
    def _load_backtest_frames(self):
        for sym in self.symbols:
            pattern = os.path.join(self.data_dir, f"{sym}_{self.interval}.csv")
            matches = glob.glob(pattern)
            if not matches:
                raise FileNotFoundError(f"No CSV for {sym} at {pattern}")

            df = pd.read_csv(matches[0], low_memory=False)

            # Expect lowercase columns from backtestingData.py:
            # ['open_time','open','high','low','close','volume','close_time','num_trades']
            cols = set(c.lower() for c in df.columns)
            needed = {"open", "high", "low", "close", "volume"}
            if not needed.issubset(cols):
                missing = needed - cols
                raise ValueError(f"{sym} file missing columns: {sorted(missing)}")

            # Timestamp source: prefer close_time; fall back to open_time.
            ts_source = "close_time" if "close_time" in cols else ("open_time" if "open_time" in cols else None)
            if ts_source is None:
                raise ValueError(f"{sym} file lacks close_time/open_time")

            # Normalize column names to lowercase once
            df.columns = [c.lower() for c in df.columns]

            # Parse datetime → integer ms (handle any stray NaT then drop)
            dt = pd.to_datetime(df[ts_source], errors="coerce", utc=True)
            ts_ms = (dt.view("int64") // 1_000_000)  # ns → ms

            out = pd.DataFrame(
                {
                    "ts": ts_ms,
                    "open": pd.to_numeric(df["open"], errors="coerce"),
                    "high": pd.to_numeric(df["high"], errors="coerce"),
                    "low": pd.to_numeric(df["low"], errors="coerce"),
                    "close": pd.to_numeric(df["close"], errors="coerce"),
                    "volume": pd.to_numeric(df["volume"], errors="coerce"),
                }
            )
            # Drop rows with bad/missing values before casting
            out = out.dropna(subset=["ts", "open", "high", "low", "close", "volume"]).copy()
            out = out.astype(
                {
                    "ts": "int64",
                    "open": "float64",
                    "high": "float64",
                    "low": "float64",
                    "close": "float64",
                    "volume": "float64",
                }
            )
            # Order & dedupe by time
            out = out.sort_values("ts")
            out = out[~out["ts"].duplicated(keep="last")].reset_index(drop=True)

            self._frames[sym] = out
            self._idx[sym] = 0

    # ------------------------------
    # Public API
    # ------------------------------
    def next_candle(self, symbol: str) -> Optional[Candle]:
        symbol = symbol.upper()
        if symbol not in self.symbols:
            raise KeyError(f"Unknown symbol: {symbol}")

        if self.mode == "backtest":
            i = self._idx[symbol]
            df = self._frames[symbol]
            if i >= len(df):
                return None
            row = df.iloc[i]
            self._idx[symbol] += 1
            return {
                "ts": int(row["ts"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }

        # live (poll latest *closed* candle)
        if self.mode == "binance":
            url = f"{self.binance_base_url}/api/v3/klines"
            params = {"symbol": symbol, "interval": self.interval, "limit": 2}
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            if not data:
                return None
            last = data[-1]
            close_time = int(last[6])
            # Only return closed candle
            if close_time >= int(time.time() * 1000) - 500:
                return None
            return {
                "ts": close_time,
                "open": float(last[1]),
                "high": float(last[2]),
                "low": float(last[3]),
                "close": float(last[4]),
                "volume": float(last[5]),
            }

        return None