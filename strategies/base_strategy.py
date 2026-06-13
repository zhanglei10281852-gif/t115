from typing import Dict
from abc import ABC, abstractmethod
from backtest.battery import BatteryState


class BaseStrategy(ABC):
    name: str = "base_strategy"

    @abstractmethod
    def decide(self, batteries: Dict[str, BatteryState], context: dict) -> Dict[str, float]:
        pass
