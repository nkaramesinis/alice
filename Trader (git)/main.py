# main.py

from trader import Trader
from momentum_strategy import MomentumStrategy
from ml_momentum_strategy import MLMomentumStrategy
from pullback_reversal_strategy import PullbackReversalStrategy

def main():
    trader = Trader(strategy=MomentumStrategy())
    trader.run_backtest()

if __name__ == "__main__":
    main()