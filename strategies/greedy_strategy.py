from typing import Dict
import numpy as np
from .base_strategy import BaseStrategy
from backtest.battery import BatteryState


class GreedyStrategy(BaseStrategy):
    name = "贪心策略"

    def __init__(self, min_spread=0.05):
        self.min_spread = min_spread

    def decide(self, batteries: Dict[str, BatteryState], context: dict) -> Dict[str, float]:
        price = context['price']
        dt_hours = context['dt_hours']
        dr_reservations = context.get('dr_reservations', {})

        commands = {}

        sorted_by_profit = []

        for bid, bat in batteries.items():
            if bat.offline:
                commands[bid] = 0
                continue

            available_charge_power = bat.get_available_charge_energy(dt_hours) / dt_hours
            available_discharge_power = bat.get_available_discharge_energy(dt_hours) / dt_hours

            reserved_power = dr_reservations.get(bid, 0)
            effective_discharge_power = max(0, available_discharge_power - reserved_power)

            if price < 0.5 and available_charge_power > 0:
                profit_potential = (0.5 - price) * available_charge_power * bat.round_trip_efficiency
                sorted_by_profit.append((profit_potential, bid, 'charge', available_charge_power))
            elif price > 0.5 and effective_discharge_power > 0:
                profit_potential = (price - 0.5) * effective_discharge_power / bat.round_trip_efficiency
                sorted_by_profit.append((profit_potential, bid, 'discharge', effective_discharge_power))

        sorted_by_profit.sort(reverse=True, key=lambda x: x[0])

        total_charge = 0
        total_discharge = 0

        for profit, bid, action, power in sorted_by_profit:
            if profit < self.min_spread * power * dt_hours:
                commands[bid] = 0
                continue

            if action == 'charge':
                commands[bid] = power
                total_charge += power
            else:
                commands[bid] = -power
                total_discharge += power

        for bid in batteries:
            if bid not in commands:
                commands[bid] = 0

        return commands
