from typing import Dict
import numpy as np
from .base_strategy import BaseStrategy
from backtest.battery import BatteryState


class RollingOptimizationStrategy(BaseStrategy):
    name = "滚动优化策略"

    def __init__(self, look_ahead_steps=96, charge_rate=0.9):
        self.look_ahead_steps = look_ahead_steps
        self.charge_rate = charge_rate

    def decide(self, batteries: Dict[str, BatteryState], context: dict) -> Dict[str, float]:
        step_idx = context['step_idx']
        market_data = context['market_data']
        dt_hours = context['dt_hours']
        dr_reservations = context.get('dr_reservations', {})

        end_idx = min(step_idx + self.look_ahead_steps, len(market_data))
        future_prices = market_data['price'].iloc[step_idx:end_idx].values

        commands = {}

        for bid, bat in batteries.items():
            if bat.offline:
                commands[bid] = 0
                continue

            reserved_power = dr_reservations.get(bid, 0)

            action, power = self._optimize_single_battery(
                bat, future_prices, dt_hours, reserved_power
            )

            if action == 'charge':
                commands[bid] = power
            elif action == 'discharge':
                commands[bid] = -power
            else:
                commands[bid] = 0

        return commands

    def _optimize_single_battery(self, bat: BatteryState, prices: np.ndarray,
                                  dt_hours: float, reserved_power: float) -> tuple:
        n = len(prices)
        if n == 0:
            return 'idle', 0

        soc = bat.soc
        soc_min = bat.soc_min
        soc_max = bat.soc_max
        capacity = bat.capacity_kwh
        max_power = bat.max_power_kw * self.charge_rate
        eta = bat.round_trip_efficiency
        deg_cost = bat.degradation_cost_per_kwh

        effective_prices = prices.copy()

        min_idx = np.argmin(effective_prices)
        max_idx = np.argmax(effective_prices)

        buy_price = effective_prices[min_idx]
        sell_price = effective_prices[max_idx]

        spread = sell_price * eta - buy_price - 2 * deg_cost

        if spread <= 0 or min_idx >= max_idx:
            return 'idle', 0

        current_price = prices[0]

        if current_price <= buy_price * 1.05 and soc < soc_max * 0.9:
            charge_power = min(max_power, bat.get_available_charge_energy(dt_hours) / dt_hours)
            return 'charge', charge_power

        if current_price >= sell_price * 0.95 and soc > soc_min + 0.1:
            available_discharge = bat.get_available_discharge_energy(dt_hours) / dt_hours
            discharge_power = min(max_power, max(0, available_discharge - reserved_power))
            if discharge_power > 0:
                return 'discharge', discharge_power

        return 'idle', 0
