import pandas as pd
import ta
from strategy_base import TradingStrategy

MAX_HOLDING_CANDLES = 5
COMMISSION_USDT = 1
PROFIT_TARGET_PCT = 0.02
STOP_LOSS_PCT = 0.02
INVESTMENT_PER_TRADE = 200

class MomentumStrategy(TradingStrategy):
    def __init__(self):
        self.df = pd.DataFrame()

    def update_df(self, candle):
        self.df = pd.concat([self.df, pd.DataFrame([candle])], ignore_index=True)
        if len(self.df) > 50:
            self.df = self.df.iloc[-50:].reset_index(drop=True)

        self.df["ema_9"] = ta.trend.ema_indicator(self.df["close"], window=9)
        self.df["ema_30"] = ta.trend.ema_indicator(self.df["close"], window=21)

    def filters_pass(self, candle, market_context) -> bool:
        if len(self.df) < 3:
            return False

        c1, c2, c3 = self.df.iloc[-3], self.df.iloc[-2], self.df.iloc[-1]

        body1 = c1['close'] - c1['open']
        if body1 >= 0:
            return False
        body2 = c2['close'] - c2['open']
        if body2 >= 0:
            return False
        body3 = c3['close'] - c3['open']
        if body3 <= 0:
            return False
        if body3 <= body2:
            return False

        vol_diff = c3['volume'] - c2['volume']
        if vol_diff <=0:
            return False

        ema_9 = self.df["ema_9"].iloc[-1]
        ema_21 = self.df["ema_21"].iloc[-1]
        if ema_9 <= ema_21:
            return False

        return True

    def trigger_fire(self, current_candle, previous_candle, market_context) -> bool:
        if len(self.df) < 2:
            return False

        c3 = self.df.iloc[-1]
        c2 = self.df.iloc[-2]

        return c3['close'] > c3['open']

    def should_enter_trade(self, candle, previous_candle, market_context) -> bool:
        self.update_df(candle)
        return self.filters_pass(candle, market_context) and self.trigger_fire(candle, previous_candle, market_context)

    def should_exit_trade(self, df):
        return False

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

        final_close = candles["close"].iloc[min(trade["start_idx"] + MAX_HOLDING_CANDLES, len(candles) - 1)]
        gain_pct = (final_close - average_entry_price) / average_entry_price
        return (gain_pct > 0), gain_pct * total_size - COMMISSION_USDT
