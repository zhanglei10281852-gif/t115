from typing import Dict
from .base_strategy import BaseStrategy
from backtest.battery import BatteryState


class ThresholdStrategy(BaseStrategy):
    name = "阈值策略"

    def __init__(self, buy_threshold=0.4, sell_threshold=0.6, charge_rate=0.8):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.charge_rate = charge_rate

    def decide(self, batteries: Dict[str, BatteryState], context: dict) -> Dict[str, float]:
        price = context['price']
        dt_hours = context['dt_hours']
        dr_reservations = context.get('dr_reservations', {})

        commands = {}

        for bid, bat in batteries.items():
            if bat.offline:
                commands[bid] = 0
                continue

            reserved_power = dr_reservations.get(bid, 0)

            if price <= self.buy_threshold:
                charge_power = bat.max_power_kw * self.charge_rate
                commands[bid] = charge_power
            elif price >= self.sell_threshold:
                discharge_power = bat.max_power_kw * self.charge_rate
                available_discharge = bat.get_available_discharge_energy(dt_hours) / dt_hours

                total_allowed = max(0, available_discharge - reserved_power)
                actual_discharge = min(discharge_power, total_allowed)
                commands[bid] = -actual_discharge
            else:
                commands[bid] = 0

        return commands
