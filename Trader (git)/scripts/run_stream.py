# scripts/run_stream.py
import os, sys, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(BASE, "src")
if SRC not in sys.path: sys.path.insert(0, SRC)

from Exchange import Exchange          # the new clean Exchange
from trader import Trader              # the clean engine you put on canvas
from ml_momentum_strategy import MLMomentumStrategy  # or your MomentumStrategy

# --- CONFIG ---
MODE       = "backtest"                 # "backtest" or "binance"
INTERVAL   = "5m"
SYMBOLS    = ["BTCUSDT"]   # add more if you like
DATA_DIR   = os.path.join(BASE, "data")          # CSV folder for backtests
SLEEP_SEC  = 1.0                                   # pacing for live mode
CAPITAL    = 1000.0
# -------------

def ema_update(prev, price, period):
    k = 2 / (period + 1)
    return price if prev is None else prev + k * (price - prev)

def main():
    ex = Exchange(
        mode=MODE,
        interval=INTERVAL,
        symbols=SYMBOLS,
        data_dir=DATA_DIR if MODE == "backtest" else None
    )
    strategy = MLMomentumStrategy(model=None)   # plug your model/thresholds if needed
    trader   = Trader(strategy=strategy, initial_capital=CAPITAL, max_positions=5)

    # Global BTC trend filter
    btc_ema9 = btc_ema21 = None

    # Track which symbols still have data (backtest) or are active (live)
    alive = set(SYMBOLS)

    while alive:
        progressed = False

        for sym in list(alive):
            candle = ex.next_candle(sym)   # << pass the symbol here
            if candle is None:
                # In backtest: this symbol is done
                # In live: just skip this turn (candle not closed yet)
                if MODE == "backtest":
                    alive.discard(sym)
                continue

            progressed = True

            # Update BTC-only market context
            if sym == "BTCUSDT":
                c = float(candle["close"])
                btc_ema9  = ema_update(btc_ema9,  c, 9)
                btc_ema21 = ema_update(btc_ema21, c, 21)

            market_context = {
                "btc_trend_up": (btc_ema9 is not None and btc_ema21 is not None and btc_ema9 > btc_ema21),
                "recovery_phase": getattr(trader, "recovery_phase", False),
            }

            trader.on_candle(sym, candle, market_context)

        if MODE == "binance":
            # live: don’t hammer the API
            time.sleep(SLEEP_SEC)
        else:
            # backtest: if nothing advanced this loop, we’re done
            if not progressed:
                break

    print(trader.summary())

if __name__ == "__main__":
    main()