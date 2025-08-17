from __future__ import annotations
from typing import Dict, Any, Optional

Candle = Dict[str, float | int]


class Trader:
    """
    Clean trading engine (no I/O, no plotting):
      - Receives one candle at a time via on_candle()
      - Delegates decision-making to a strategy object
      - Manages positions, PnL, and a per-symbol state bag

    Strategy contract (example):
      strategy.on_candle(symbol: str, candle: Candle, market_context: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]
      where the returned dict may include keys like:
        {"open_long": bool, "close_long": bool, "size": float, "tp": float, "sl": float}
    """

    def __init__(self, strategy, initial_capital: float, max_positions: int = 5):
        self.strategy = strategy
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.max_positions = int(max_positions)

        # Open positions by symbol
        # schema: {symbol: {"entry": float, "size": float, "tp": Optional[float], "sl": Optional[float]}}
        self.positions: Dict[str, Dict[str, Any]] = {}

        # Per-symbol rolling state for indicators, counters, etc. (owned by the engine, used by strategies)
        self.state: Dict[str, Dict[str, Any]] = {}

        # Trade log for realized events
        self.trades: list[Dict[str, Any]] = []

        # Optional global flags used by your strategies (kept for backwards-compatibility)
        self.recovery_phase: bool = False

    # ------------------------------
    # Public API
    # ------------------------------
    def on_candle(self, symbol: str, candle: Candle, market_context: Dict[str, Any]):
        """Process a single candle for a symbol.
        The engine stays O(1) per candle; heavy feature calc belongs in strategy/state.
        """
        st = self.state.setdefault(symbol, {})

        # 1) Strategy decides actions
        actions = self.strategy.on_candle(symbol, candle, market_context, st) or {}

        # 2) Optional: update per-position risk controls before acting (TSL, time-based exits, etc.)
        self._risk_manage_open_position(symbol, candle)

        # 3) Execute actions in deterministic order: close → open → modify
        if actions.get("close_long"):
            self._close_long(symbol, candle)

        if actions.get("open_long"):
            size = float(actions.get("size", 1.0))
            tp = _safe_float(actions.get("tp"))
            sl = _safe_float(actions.get("sl"))
            self._open_long(symbol, candle, size=size, tp=tp, sl=sl)

        if "tp" in actions or "sl" in actions:
            # post-open modification (e.g., trail stop after entry)
            self._modify_orders(symbol, tp=_safe_float(actions.get("tp")), sl=_safe_float(actions.get("sl")))

    def summary(self) -> Dict[str, Any]:
        realized = sum(t.get("pnl", 0.0) for t in self.trades if t.get("type") == "SELL")
        return {
            "initial_capital": self.initial_capital,
            "cash": round(self.cash, 6),
            "open_positions": len(self.positions),
            "trades": len(self.trades),
            "realized_pnl": round(realized, 6),
        }

    # ------------------------------
    # Internals
    # ------------------------------
    def _open_long(self, symbol: str, candle: Candle, *, size: float, tp: Optional[float], sl: Optional[float]):
        if symbol in self.positions:
            return  # already long; ignore or convert to scale-in logic
        if len(self.positions) >= self.max_positions:
            return
        price = float(candle["close"])
        self.positions[symbol] = {"entry": price, "size": float(size), "tp": tp, "sl": sl}
        self.trades.append({
            "type": "BUY",
            "symbol": symbol,
            "price": price,
            "size": float(size),
            "ts": int(candle["ts"]),
        })

    def _close_long(self, symbol: str, candle: Candle):
        pos = self.positions.pop(symbol, None)
        if not pos:
            return
        exit_p = float(candle["close"])
        pnl = (exit_p - float(pos["entry"])) * float(pos["size"])
        self.cash += pnl
        self.trades.append({
            "type": "SELL",
            "symbol": symbol,
            "price": exit_p,
            "size": float(pos["size"]),
            "ts": int(candle["ts"]),
            "pnl": pnl,
        })

    def _modify_orders(self, symbol: str, *, tp: Optional[float], sl: Optional[float]):
        if symbol not in self.positions:
            return
        if tp is not None:
            self.positions[symbol]["tp"] = float(tp)
        if sl is not None:
            self.positions[symbol]["sl"] = float(sl)

    def _risk_manage_open_position(self, symbol: str, candle: Candle):
        """Simple built-ins: take-profit / stop-loss execution. Extend as needed."""
        pos = self.positions.get(symbol)
        if not pos:
            return
        high = float(candle["high"]) if "high" in candle else float(candle["close"])
        low = float(candle["low"]) if "low" in candle else float(candle["close"])
        price_for_tp = high
        price_for_sl = low

        # Check TP
        tp = pos.get("tp")
        if tp is not None and price_for_tp >= float(tp):
            # simulate fill at TP
            fake_candle = dict(candle)
            fake_candle["close"] = float(tp)
            self._close_long(symbol, fake_candle)
            return  # position is gone

        # Check SL
        sl = pos.get("sl")
        if sl is not None and price_for_sl <= float(sl):
            # simulate fill at SL
            fake_candle = dict(candle)
            fake_candle["close"] = float(sl)
            self._close_long(symbol, fake_candle)


def _safe_float(x: Any) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None
