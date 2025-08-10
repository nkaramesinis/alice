# strategy_base.py

from abc import ABC, abstractmethod

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

