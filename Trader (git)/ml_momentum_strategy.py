# ml_momentum_strategy.py

import joblib
import pandas as pd
import numpy as np
from strategy_base import TradingStrategy

class MLMomentumStrategy(TradingStrategy):
    def __init__(self):
        # Load trained ML model
        self.model = joblib.load("momentum_model.pkl")

    def filters_pass(self, candle, market_context) -> bool:
        try:
            proba = candle.get("ml_proba", 0)
            #print(f"Probability of profit: {proba:.2f}")
            return proba > 0.45  # adjust as needed
        except Exception as e:
            print(f"[ERROR in filters_pass]: {e}")
            return False


    #def filters_pass(self, candle, market_context) -> bool:
     #   try:
      #      features = {
       #         "ema_gap": (candle["ema_9"] - candle["ema_21"]) / candle["ema_21"],
        #        "rsi": candle["rsi"],
         #       "volume_ratio": candle["volume"] / candle["avg_volume"] if candle["avg_volume"] > 0 else 1,
          #      "bb_width": candle["bb_high"] - candle["bb_low"],
           #     "above_ema_50": int(candle["close"] > candle["ema_50"])
            #}

        #    X = pd.DataFrame([features])
         #   proba = self.model.predict_proba(X)[0][1]  # Probability of class 1 (profitable)

          #  print(f"Probability of profit: {proba:.2f}")
            #print(f"[ML] Features: {features}, Probability of profit: {proba:.2f}")
          #  return proba > 0.6  # Adjust threshold as needed

      #  except Exception as e:
       #     print(f"[ERROR in filters_pass]: {e}")
       #     return False

    def trigger_fire(self, current_candle, previous_candle, market_context) -> bool:
        return current_candle["close"] #> previous_candle["close"]

    def should_enter_trade(self, candle, previous_candle, market_context) -> bool:
        return self.filters_pass(candle, market_context) #and self.trigger_fire(candle, previous_candle, market_context)

    def manage_open_trade(self, trade, candles) -> (bool, float):
        # Same trade management logic as before
        entry_price = trade["entry_price"]
        total_size = trade["size"]
        average_entry_price = entry_price
        PROFIT_TARGET_PCT = 0.03
        STOP_LOSS_PCT = 0.01
        COMMISSION_USDT = 1
        MAX_HOLDING_CANDLES = 15

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
                average_entry_price = (average_entry_price * total_size + close * 200) / (total_size + 200)
                total_size += 200

            if candles["ema_9"].iloc[i] < candles["ema_21"].iloc[i]:
                gain_pct = (close - average_entry_price) / average_entry_price
                return (gain_pct > 0), gain_pct * total_size - COMMISSION_USDT

        final_close = candles["close"].iloc[min(trade["start_idx"] + MAX_HOLDING_CANDLES, len(candles) - 1)]
        gain_pct = (final_close - average_entry_price) / average_entry_price
        return (gain_pct > 0), gain_pct * total_size - COMMISSION_USDT
