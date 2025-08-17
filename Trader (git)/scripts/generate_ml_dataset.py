# scripts/generate_ml_dataset.py
import os
from pathlib import Path
import pandas as pd
import ta

# ===== Paths (repo-relative, robust) =====
SCRIPT_DIR = Path(__file__).resolve().parent            # .../Trader (git)/scripts
REPO_ROOT  = SCRIPT_DIR.parent                          # .../Trader (git)
DATA_ROOT  = (REPO_ROOT / "data").resolve()             # .../Trader (git)/data
OUTPUT_FILE = (REPO_ROOT / "ml_dataset.csv").resolve()  # write dataset at repo root

# ===== Labeling params =====
PROFIT_TARGET_PCT = 0.02
STOP_LOSS_PCT = 0.01
MAX_HOLDING_CANDLES = 60


def simulate_trade(entry_idx: int, df: pd.DataFrame) -> int:
    entry_price = float(df.loc[entry_idx, "close"])
    end_idx = min(entry_idx + MAX_HOLDING_CANDLES, len(df) - 1)
    for i in range(entry_idx + 1, end_idx + 1):
        high = float(df.loc[i, "high"])
        low  = float(df.loc[i, "low"])
        if high >= entry_price * (1 + PROFIT_TARGET_PCT):
            return 1  # profit hit before stop
        if low  <= entry_price * (1 - STOP_LOSS_PCT):
            return 0  # stop loss hit first
    return 0  # target not hit within holding window


def process_file(file_path: Path, ticker: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, low_memory=False)

    # Expect lowercase cols from downloader:
    # ['open_time','open','high','low','close','volume','close_time','num_trades']
    cols = {c.lower(): c for c in df.columns}
    need = ["open", "high", "low", "close", "volume"]
    for n in need:
        if n not in cols:
            raise ValueError(f"{ticker}: missing column '{n}' in {file_path.name}")

    # Choose timestamp source: prefer close_time, fallback open_time
    ts_col = None
    if "close_time" in cols: ts_col = cols["close_time"]
    elif "open_time" in cols: ts_col = cols["open_time"]
    else:
        raise ValueError(f"{ticker}: no close_time/open_time column in {file_path.name}")

    # Normalize types
    price_cols = [cols["open"], cols["high"], cols["low"], cols["close"], cols["volume"]]
    df[price_cols] = df[price_cols].apply(pd.to_numeric, errors="coerce")

    # Build a 'timestamp' column (datetime) and sort
    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"] + price_cols).sort_values("timestamp").reset_index(drop=True)

    # Indicators (match training/strategy features)
    df["ema_9"]  = ta.trend.ema_indicator(df[cols["close"]], window=9)
    df["ema_21"] = ta.trend.ema_indicator(df[cols["close"]], window=21)
    df["ema_50"] = ta.trend.ema_indicator(df[cols["close"]], window=50)
    df["rsi"]    = ta.momentum.RSIIndicator(df[cols["close"]], window=14).rsi()
    df["avg_volume"] = df[cols["volume"]].rolling(window=14).mean()

    bb = ta.volatility.BollingerBands(df[cols["close"]], window=20, window_dev=2)
    df["bb_high"]  = bb.bollinger_hband()
    df["bb_low"]   = bb.bollinger_lband()
    df["bb_width"] = df["bb_high"] - df["bb_low"]

    # Drop warmup NaNs from indicators
    df = df.dropna(subset=["ema_9","ema_21","ema_50","rsi","avg_volume","bb_high","bb_low","bb_width"]).reset_index(drop=True)

    # Build rows with features + label
    rows = []
    last_entry = len(df) - MAX_HOLDING_CANDLES - 1
    for i in range(max(50, 0), max(last_entry, 0)):
        row = df.iloc[i]
        label = simulate_trade(i, df)
        ema_gap = (float(row["ema_9"]) - float(row["ema_21"]))
        denom   = float(row["ema_21"]) if abs(float(row["ema_21"])) > 1e-12 else 1.0
        avg_vol = float(row["avg_volume"]) if float(row["avg_volume"]) > 0 else 1.0

        rows.append({
            "ticker": ticker,
            "timestamp": row["timestamp"],  # keep as datetime; writer will handle ISO
            "close": float(row[cols["close"]]),
            "ema_gap": ema_gap / denom,
            "rsi": float(row["rsi"]),
            "volume_ratio": float(row[cols["volume"]]) / avg_vol,
            "bb_width": float(row["bb_width"]),
            "above_ema_50": 1 if float(row[cols["close"]]) > float(row["ema_50"]) else 0,
            "label": int(label),
        })

    return pd.DataFrame(rows)


def main():
    datasets = []
    files = list(DATA_ROOT.rglob("*_5m.csv"))  # recurse data/** for *_5m.csv
    if not files:
        print(f"No 5m csv files found under {DATA_ROOT}")
        return

    for fp in files:
        ticker = fp.name.replace("_5m.csv", "").upper()
        try:
            df = process_file(fp, ticker)
            if not df.empty:
                datasets.append(df)
                print(f"[OK] {ticker}: +{len(df)} rows from {fp}")
            else:
                print(f"[SKIP] {ticker}: no rows after indicator warmup")
        except Exception as e:
            print(f"[ERR] {ticker} in {fp}: {e}")

    if not datasets:
        print("No data accumulated; nothing to write.")
        return

    all_data = pd.concat(datasets, ignore_index=True)
    all_data.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset saved to {OUTPUT_FILE} with {len(all_data)} rows.")


if __name__ == "__main__":
    main()
