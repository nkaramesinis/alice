import pandas as pd
import os
import ta
from ta.volatility import BollingerBands
from strategy_base import TradingStrategy
from joblib import load
import numpy as np
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = "../../backtest_data"
BTC_FILE = "BTCUSDT_5m.csv"
INVESTMENT_PER_TRADE = 200
MAX_CAPITAL = 1000
COMMISSION_USDT = 1
MAX_HOLDING_CANDLES = 15
model = load("../momentum_model.pkl")

class Trader:
    def __init__(self, strategy: TradingStrategy):
        self.strategy = strategy
        self.total_profit = 0
        self.total_trades = 0
        self.successful_trades = 0
        self.positions = []
        self.recent_losses = []
        self.trading_paused_until = None
        self.recovery_phase = 3  # 1: 20%, 2: 50%, 3: 100%
        self.current_capital = MAX_CAPITAL
        self.recent_candles = []
        self.trade_signals = []  # Store trade signals

    def trade(self, candle, market_context) -> int:
        self.recent_candles.append(candle)
        if len(self.recent_candles) < 2:
            return 0  # Not enough data
        if len(self.recent_candles) > 50:
            self.recent_candles.pop(0)

        previous_candle = self.recent_candles[-2]
        if self.strategy.should_enter_trade(candle, previous_candle, market_context):
            return 1  # Buy signal
        return 0  # No signal

    def load_data(self):
        dfs = []
        for file in os.listdir(DATA_DIR):
            if file.endswith("5m.csv"):
                symbol = file.replace("_5m.csv", "")
                df = pd.read_csv(os.path.join(DATA_DIR, file))
                df["ticker"] = symbol
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["close"] = df["close"].astype(float)
                df["open"] = df["open"].astype(float)
                df["high"] = df["high"].astype(float)
                df["low"] = df["low"].astype(float)
                df["volume"] = df["volume"].astype(float)

                df["ema_9"] = ta.trend.ema_indicator(df["close"], window=9)
                df["ema_21"] = ta.trend.ema_indicator(df["close"], window=21)
                df["ema_50"] = ta.trend.ema_indicator(df["close"], window=50)
                df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

                bb = BollingerBands(close=df["close"], window=20, window_dev=2)
                df["bb_high"] = bb.bollinger_hband()
                df["bb_low"] = bb.bollinger_lband()
                df["bb_width"] = df["bb_high"] - df["bb_low"]

                df["avg_volume"] = df["volume"].rolling(window=14).mean()

                # Compute model features
                df["ema_gap"] = (df["ema_9"] - df["ema_21"]) / df["ema_21"]
                df["volume_ratio"] = df["volume"] / df["avg_volume"]
                df["above_ema_50"] = (df["close"] > df["ema_50"]).astype(int)
                df["bb_width"] = df["bb_high"] - df["bb_low"]

                # Prepare model input
                features = ["ema_gap", "rsi", "volume_ratio", "bb_width", "above_ema_50"]
                df_model = df[features].copy()
                df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna()

                # Predict probabilities in batch
                df.loc[df_model.index, "ml_proba"] = model.predict_proba(df_model)[:, 1]

                dfs.append(df)
        return pd.concat(dfs).sort_values(by="timestamp").reset_index(drop=True)

    def load_btc_trend(self):
        btc_path = os.path.join(DATA_DIR, BTC_FILE)
        df_btc = pd.read_csv(btc_path)
        df_btc["timestamp"] = pd.to_datetime(df_btc["timestamp"])
        df_btc["close"] = df_btc["close"].astype(float)
        df_btc["ema_9"] = ta.trend.ema_indicator(df_btc["close"], window=9)
        df_btc["ema_21"] = ta.trend.ema_indicator(df_btc["close"], window=21)
        df_btc["btc_trend_up"] = (
            (df_btc["close"] > df_btc["ema_9"]) &
            (df_btc["ema_9"] > df_btc["ema_21"])
        )
        return df_btc[["timestamp", "btc_trend_up"]]

    def run_backtest(self):
        df = self.load_data()
        btc_trend = self.load_btc_trend()
        df = df.merge(btc_trend, on="timestamp", how="left")

        all_timestamps = df["timestamp"].drop_duplicates().sort_values().reset_index(drop=True)

        for current_time in all_timestamps:
            if self.trading_paused_until and current_time < self.trading_paused_until:
                continue
            if self.trading_paused_until and current_time >= self.trading_paused_until:
                self.recovery_phase = 1
                self.trading_paused_until = None

            self.positions = [pos for pos in self.positions if pos["exit_time"] > current_time]
            #print(f"Timestamp: {current_time}")

            if self.recovery_phase == 1:
                cap_limit = 0.2 * self.current_capital
            elif self.recovery_phase == 2:
                cap_limit = 0.5 * self.current_capital
            else:
                cap_limit = self.current_capital

            open_positions = [pos for pos in self.positions if pos["exit_time"] > current_time]
            #print (open_positions)
            capital_used = sum(pos["capital"] for pos in open_positions)
            #print ("capital used: ", capital_used)
            capital_available = cap_limit - capital_used
            current_candles = df[df["timestamp"] == current_time]

            for _, row in current_candles.iterrows():
                ticker_data = df[df["ticker"] == row["ticker"]].reset_index(drop=True)
                idx = ticker_data[ticker_data["timestamp"] == current_time].index[0]

                if idx < 50 or idx + 1 >= len(ticker_data):
                    continue

                candle = ticker_data.iloc[idx]
                previous_candle = ticker_data.iloc[idx - 1]
                market_context = {
                    "btc_trend_up": row.get("btc_trend_up", True),
                    "recovery_phase": self.recovery_phase
                }

                if self.strategy.should_enter_trade(candle, previous_candle, market_context):
                    if capital_available >= INVESTMENT_PER_TRADE:
                        trade = {"entry_price": candle["close"], "size": INVESTMENT_PER_TRADE, "start_idx": idx}
                        success, profit = self.strategy.manage_open_trade(trade, ticker_data)
                        capital_available -= INVESTMENT_PER_TRADE
                        self.positions.append({
                            "ticker": row["ticker"],
                            "entry_time": current_time,
                            "exit_time": current_time + pd.Timedelta(minutes=5 * MAX_HOLDING_CANDLES),
                            "capital": INVESTMENT_PER_TRADE
                        })

                        # Record signal
                        self.trade_signals.append({
                            "ticker": row["ticker"],
                            "timestamp": current_time,
                            "price": candle["close"],
                            "successful": success
                        })

                        print(f"Open position {row['ticker']} at {current_time} and profit was {profit:.2f} and total profit {self.total_profit:.2f} and capital available: {capital_available:.2f}")

                        self.total_profit += profit
                        self.current_capital = MAX_CAPITAL + self.total_profit
                        self.total_trades += 1
                        if success:
                            self.successful_trades += 1
                            if self.recovery_phase == 1:
                                self.recovery_phase = 2
                            elif self.recovery_phase == 2:
                                self.recovery_phase = 3
                        else:
                            self.recent_losses.append(current_time)
                            self.recent_losses = [t for t in self.recent_losses if (current_time - t).total_seconds() <= 1800]
                            if len(self.recent_losses) >= 3:
                                self.trading_paused_until = current_time + pd.Timedelta(hours=1)
                                print(f"Trading paused until {self.trading_paused_until} due to loss streak")

        self.report()
        self.plot_price_trends(df)

    def report(self):
        print(f"Total Trades: {self.total_trades}")
        print(f"Successful Trades: {self.successful_trades}")
        if self.total_trades > 0:
            print(f"Success Rate: {self.successful_trades / self.total_trades:.2%}")
            print(f"Total Profit: ${self.total_profit:.2f}")
            print(f"ROI: {(self.total_profit / MAX_CAPITAL) * 100:.2f}%")

    def plot_price_trends(self, df):
        tickers = df["ticker"].unique()
        signals_df = pd.DataFrame(self.trade_signals)
        fig, axs = plt.subplots(len(tickers), 1, figsize=(12, len(tickers) * 2), sharex=True)
        if len(tickers) == 1:
            axs = [axs]

        for ax, ticker in zip(axs, tickers):
            sub_df = df[df["ticker"] == ticker]
            ax.plot(sub_df["timestamp"], sub_df["close"], label=f"{ticker} Price")

            # Plot trade signals
            if not signals_df.empty:
                buy_signals = signals_df[(signals_df["ticker"] == ticker)]
                ax.scatter(buy_signals["timestamp"], buy_signals["price"],
                           c=buy_signals["successful"].map({True: "green", False: "red"}),
                           label="Trade Signals", marker="o", s=40, alpha=0.8)

            ax.set_title(f"{ticker} Price Trend")
            ax.set_ylabel("Price")
            ax.legend()

        plt.tight_layout()
        plt.show()
