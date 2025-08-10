# trader_architecture.py
import pandas as pd
import os
import ta
from abc import ABC, abstractmethod

# Configuration
DATA_DIR = "../backtest_data"
BTC_FILE = "BTCUSDT_5m.csv"
PROFIT_TARGET_PCT = 0.03
STOP_LOSS_PCT = 0.01
INVESTMENT_PER_TRADE = 200
MAX_CAPITAL = 1000
COMMISSION_USDT = 1
MAX_HOLDING_CANDLES = 15


class TradingStrategy(ABC):
    @abstractmethod
    def filters_pass(self, candle, market_context) -> bool:
        pass

    @abstractmethod
    def trigger_fire(self, candle, market_context) -> bool:
        pass

    @abstractmethod
    def should_enter_trade(self, candle, market_context) -> bool:
        pass

    @abstractmethod
    def manage_open_trade(self, trade, candles) -> (bool, float):
        pass


class MomentumStrategy(TradingStrategy):

    def __init__(self):
        self.levels = {
            "normal": {
                "ema_gap": 0.005,
                "rsi_threshold": 30,
                "volume_multiplier": 1.0
            },
            "strict": {
                "ema_gap": 0.01,
                "rsi_threshold": 40,
                "volume_multiplier": 1.5
            },
            "strictest": {
                "ema_gap": 0.015,
                "rsi_threshold": 45,
                "volume_multiplier": 1.5
            }
        }

    def filters_pass(self, candle, market_context) -> bool:

        recovery_phase = market_context.get("recovery_phase", 3)

        if recovery_phase == 1:
            strictness = "strictest"
        elif recovery_phase == 2:
            strictness = "strict"
        else:
            strictness = "normal"

        params = self.levels[strictness]

        ema_gap = (candle["ema_9"] - candle["ema_21"])/candle["ema_21"]

        return (
            candle["close"] > candle["ema_50"] and
            ema_gap >= params["ema_gap"] and
            candle["rsi"] > params["rsi_threshold"] and
            market_context.get("btc_trend_up", True)
        )

    def trigger_fire(self, current_candle, previous_candle, market_context) -> bool:
        return current_candle["close"] > previous_candle["high"]

    def should_enter_trade(self, candle, previous_candle, market_context) -> bool:
        return self.filters_pass(candle, market_context) and self.trigger_fire(candle, previous_candle, market_context)

    def manage_open_trade(self, trade, candles) -> (bool, float):
        entry_price = trade["entry_price"]
        total_size = trade["size"]
        average_entry_price = entry_price

        for i in range(trade["start_idx"] + 1, min(trade["start_idx"] + MAX_HOLDING_CANDLES + 1, len(candles))):
            high = candles["high"].iloc[i]
            low = candles["low"].iloc[i]
            close = candles["close"].iloc[i]

            if high >= entry_price * (1 + PROFIT_TARGET_PCT):
                gain_pct = (high - average_entry_price) / average_entry_price
                return True, gain_pct * total_size - COMMISSION_USDT

            if low <= entry_price * (1 - STOP_LOSS_PCT):
                loss_pct = (average_entry_price - low) / average_entry_price
                return False, -loss_pct * total_size - COMMISSION_USDT

            if close > average_entry_price and candles["volume"].iloc[i] > candles["avg_volume"].iloc[i]:
                average_entry_price = (average_entry_price * total_size + close * INVESTMENT_PER_TRADE) / (total_size + INVESTMENT_PER_TRADE)
                total_size += INVESTMENT_PER_TRADE

            if candles["ema_9"].iloc[i] < candles["ema_21"].iloc[i]:
                gain_pct = (close - average_entry_price) / average_entry_price
                return (gain_pct > 0), gain_pct * total_size - COMMISSION_USDT

        final_close = candles["close"].iloc[min(trade["start_idx"] + MAX_HOLDING_CANDLES, len(candles) - 1)]
        gain_pct = (final_close - average_entry_price) / average_entry_price
        return (gain_pct > 0), gain_pct * total_size - COMMISSION_USDT

        final_close = candles["close"].iloc[min(trade["start_idx"] + MAX_HOLDING_CANDLES, len(candles) - 1)]
        gain_pct = (final_close - average_entry_price) / average_entry_price
        return (gain_pct > 0), gain_pct * total_size - COMMISSION_USDT


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
        self.strategy = strategy
        self.total_profit = 0
        self.total_trades = 0
        self.successful_trades = 0
        self.positions = []

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
                df["avg_volume"] = df["volume"].rolling(window=14).mean()

                dfs.append(df)
        return pd.concat(dfs).sort_values(by="timestamp").reset_index(drop=True)

    def load_btc_trend(self):
        btc_path = os.path.join(DATA_DIR, BTC_FILE)
        df_btc = pd.read_csv(btc_path)
        df_btc["timestamp"] = pd.to_datetime(df_btc["timestamp"])
        df_btc["close"] = df_btc["close"].astype(float)
        df_btc["ema_9"] = ta.trend.ema_indicator(df_btc["close"], window=9)
        df_btc["ema_21"] = ta.trend.ema_indicator(df_btc["close"], window=21)
        df_btc["ema_50"] = ta.trend.ema_indicator(df_btc["close"], window=50)
        df_btc["rsi"] = ta.momentum.RSIIndicator(df_btc["close"], window=14).rsi()
        df_btc["btc_trend_up"] = (
            (df_btc["close"] > df_btc["ema_9"]) &
            (df_btc["ema_9"] > df_btc["ema_21"])
            #(df_btc["rsi"] > 50)
        )
        return df_btc[["timestamp", "btc_trend_up"]]

    def run_backtest(self):
        df = self.load_data()
        btc_trend = self.load_btc_trend()
        df = df.merge(btc_trend, on="timestamp", how="left")

        all_timestamps = df["timestamp"].drop_duplicates().sort_values().reset_index(drop=True)

        for current_time in all_timestamps:
            # Handle loss streak pause
            if self.trading_paused_until and current_time < self.trading_paused_until:
                continue
            if self.trading_paused_until and current_time >= self.trading_paused_until:
                self.recovery_phase = 1
                self.trading_paused_until = None

            # Clean up expired positions
            self.positions = [pos for pos in self.positions if pos["exit_time"] > current_time]
            print(f"Timestamp: {current_time}")

            # Calculate available capital dynamically
            if self.recovery_phase == 1:
                cap_limit = 0.2 * self.current_capital
            elif self.recovery_phase == 2:
                cap_limit = 0.5 * self.current_capital
            else:
                cap_limit = self.current_capital
            capital_used = sum(pos["capital"] for pos in self.positions)
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

                if capital_available >= INVESTMENT_PER_TRADE and self.strategy.should_enter_trade(candle, previous_candle, market_context):
                    trade = {"entry_price": candle["close"], "size": INVESTMENT_PER_TRADE, "start_idx": idx}
                    success, profit = self.strategy.manage_open_trade(trade, ticker_data)
                    capital_available -= INVESTMENT_PER_TRADE
                    self.positions.append({
                        "ticker": row["ticker"],
                        "entry_time": current_time,
                        "exit_time": current_time + pd.Timedelta(minutes=5 * MAX_HOLDING_CANDLES),
                        "capital": INVESTMENT_PER_TRADE
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

    def report(self):
        print(f"Total Trades: {self.total_trades}")
        print(f"Successful Trades: {self.successful_trades}")
        if self.total_trades > 0:
            print(f"Success Rate: {self.successful_trades / self.total_trades:.2%}")
            print(f"Total Profit: ${self.total_profit:.2f}")
            print(f"ROI: {(self.total_profit / MAX_CAPITAL) * 100:.2f}%")


def main():
    trader = Trader(strategy=MomentumStrategy())
    trader.run_backtest()


if __name__ == "__main__":
    main()
