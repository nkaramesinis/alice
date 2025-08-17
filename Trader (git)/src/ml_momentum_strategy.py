# src/ml_momentum_strategy.py
from __future__ import annotations
from typing import Dict, Any, Optional

import os
import joblib
import pandas as pd
import numpy as np

from strategy_base import TradingStrategy

Candle = Dict[str, float | int]


class MLMomentumStrategy(TradingStrategy):
    """
    Clean strategy that works with the new Trader:
      - Tries to use an ML model if available (or ml_proba already on the candle)
      - Falls back to a simple momentum trigger if not
      - Returns action dicts the Trader understands
    """

    def __init__(self, model_path: str = "../momentum_model.pkl", proba_threshold: float = 0.45):
        self.model = None
        self.model_path = model_path
        self.proba_threshold = float(proba_threshold)

        # Try loading a trained model; be graceful if missing
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
        except Exception as e:
            print(f"[MLMomentumStrategy] Could not load model at {self.model_path}: {e}")
            self.model = None

    # ------------------------------
    # Public API expected by Trader
    # ------------------------------
    def on_candle(self, symbol: str, candle: Candle, market_context: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return actions like:
          {"open_long": True, "size": 1.0}
          {"close_long": True}
          {"tp": 1.02 * entry, "sl": 0.99 * entry}
        Minimal example below: open if ML / momentum says so, exit via Trader's risk or simple fallback.
        """
        close = float(candle["close"])

        # keep simple per-symbol memory
        prev_close = state.get("prev_close")
        bars_held = state.get("bars_held", 0)

        # --- ENTRY LOGIC ---
        enter_long = False

        # (A) If candle already carries ml_proba, use it (your file used 0.45)
        ml_proba = _safe_float(candle.get("ml_proba"))
        if ml_proba is not None:
            if ml_proba > self.proba_threshold and market_context.get("btc_trend_up", True):
                enter_long = True

        # (B) Else, if we have a loaded model and features are available, compute proba (optional)
        # NOTE: to enable this, add your features to the candle dict (ema_9, ema_21, rsi, avg_volume, bb_high, bb_low, etc.)
        elif self.model is not None and self._has_min_features(candle):
            feats = self._build_features(candle)
            try:
                proba = float(self.model.predict_proba(feats)[0][1])
                if proba > self.proba_threshold and market_context.get("btc_trend_up", True):
                    enter_long = True
            except Exception as e:
                print(f"[MLMomentumStrategy] predict_proba failed: {e}")

        # (C) Fallback: tiny momentum heuristic if no model/proba
        elif prev_close is not None:
            if (close > float(prev_close)) and market_context.get("btc_trend_up", True):
                enter_long = True

        # --- EXIT LOGIC (simple; you can make this richer) ---
        # Let Trader's TP/SL handle most exits. As a minimal safety,
        # if price dips below previous close, suggest close.
        close_long = False
        if prev_close is not None and close < float(prev_close):
            close_long = True

        # Update state
        state["prev_close"] = close
        state["bars_held"] = bars_held + 1 if "bars_held" in state else (1 if enter_long else 0)

        # Return actions; keep it minimal and let Trader risk management do TP/SL if you set them
        actions: Dict[str, Any] = {}
        if enter_long:
            actions["open_long"] = True
            actions["size"] = 1.0  # tune sizing as you like
            # Optional: set initial TP/SL here, e.g. 3% target / 1% stop:
            # actions["tp"] = close * 1.03
            # actions["sl"] = close * 0.99

        if close_long:
            actions["close_long"] = True

        return actions

    # ------------------------------
    # Helpers
    # ------------------------------
    def _has_min_features(self, candle: Dict[str, Any]) -> bool:
        # minimal example; extend if you want the richer block you commented before
        needed = ["ema_9", "ema_21", "rsi", "volume", "avg_volume", "bb_high", "bb_low", "close"]
        return all(k in candle for k in needed)

    def _build_features(self, candle: Dict[str, Any]) -> pd.DataFrame:
        ema9 = float(candle["ema_9"])
        ema21 = float(candle["ema_21"])
        close = float(candle["close"])
        volume = float(candle["volume"])
        avg_volume = max(float(candle["avg_volume"]), 1e-9)
        bb_high = float(candle["bb_high"])
        bb_low = float(candle["bb_low"])
        feats = {
            "ema_gap": (ema9 - ema21) / (ema21 if abs(ema21) > 1e-9 else 1.0),
            "rsi": float(candle["rsi"]),
            "volume_ratio": volume / avg_volume,
            "bb_width": bb_high - bb_low,
            "above_ema_50": float(candle.get("close", close) > float(candle.get("ema_50", ema21))),
        }
        return pd.DataFrame([feats])


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
