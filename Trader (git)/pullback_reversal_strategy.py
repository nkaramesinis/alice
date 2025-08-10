# pullback_reversal_strategy.py

import numpy as np
from strategy_base import TradingStrategy

# Configuration
PROFIT_TARGET_PCT = 0.03
STOP_LOSS_PCT = 0.01
INVESTMENT_PER_TRADE = 200
MAX_CAPITAL = 1000
COMMISSION_USDT = 1
MAX_HOLDING_CANDLES = 15

class PullbackReversalStrategy(TradingStrategy):
    def filters_pass(self, candle, market_context) -> bool:
        # Uptrend condition
        return candle["ema_9"] > candle["ema_21"]

    def detect_pullback(self, previous_candles) -> bool:
        red_candles = [c for c in previous_candles if c["close"] < c["open"]]
        return 3 <= len(red_candles) <= 5

    def body_size(self, candle) -> float:
        return abs(candle["close"] - candle["open"])

    def average_volume(self, candles) -> float:
        volumes = [c["volume"] for c in candles]
        return np.mean(volumes)

    def trigger_fire(self, current_candle, previous_candles, market_context) -> bool:
        if not self.detect_pullback(previous_candles):
            return False
        return (
            current_candle["close"] > current_candle["open"] and  # Green candle
            self.body_size(current_candle) > self.body_size(previous_candles[-1]) and
            current_candle["volume"] > self.average_volume(previous_candles)
        )

    def should_enter_trade(self, current_candle, previous_candles, market_context) -> bool:
        return self.filters_pass(current_candle, market_context) and self.trigger_fire(current_candle, previous_candles, market_context)

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
