import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable


@dataclass
class BatteryState:
    battery_id: str
    capacity_kwh: float
    max_power_kw: float
    soc: float
    soc_min: float
    soc_max: float
    round_trip_efficiency: float
    degradation_cost_per_kwh: float
    offline: bool = False
    total_energy_charged: float = 0.0
    total_energy_discharged: float = 0.0
    total_degradation_cost: float = 0.0
    arbitrage_revenue: float = 0.0
    dr_revenue: float = 0.0
    reg_revenue: float = 0.0
    soc_history: List[float] = field(default_factory=list)
    power_history: List[float] = field(default_factory=list)
    status_history: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def charge(self, power_kw: float, dt_hours: float, price: float) -> dict:
        if self.offline:
            return {'actual_power': 0, 'energy': 0, 'cost': 0, 'degradation': 0, 'status': 'offline'}

        power_kw = max(0, min(power_kw, self.max_power_kw))
        energy_charge = power_kw * dt_hours

        max_energy_charge = (self.soc_max - self.soc) * self.capacity_kwh / self.round_trip_efficiency

        if energy_charge > max_energy_charge:
            energy_charge = max_energy_charge
            power_kw = energy_charge / dt_hours if dt_hours > 0 else 0
            self.warnings.append(f"充电触达SOC上限 {self.soc_max:.2f}")

        actual_energy_stored = energy_charge * self.round_trip_efficiency
        self.soc += actual_energy_stored / self.capacity_kwh
        self.total_energy_charged += energy_charge

        cost = energy_charge * price
        degradation_cost = energy_charge * self.degradation_cost_per_kwh
        self.total_degradation_cost += degradation_cost
        self.arbitrage_revenue -= cost
        self.arbitrage_revenue -= degradation_cost

        self.soc_history.append(self.soc)
        self.power_history.append(power_kw)
        self.status_history.append('charging')

        return {
            'actual_power': power_kw,
            'energy': energy_charge,
            'cost': cost,
            'degradation': degradation_cost,
            'status': 'charging',
        }

    def discharge(self, power_kw: float, dt_hours: float, price: float) -> dict:
        if self.offline:
            return {'actual_power': 0, 'energy': 0, 'revenue': 0, 'degradation': 0, 'status': 'offline'}

        power_kw = max(0, min(power_kw, self.max_power_kw))
        energy_discharge = power_kw * dt_hours

        available_energy = (self.soc - self.soc_min) * self.capacity_kwh * self.round_trip_efficiency

        if energy_discharge > available_energy:
            energy_discharge = available_energy
            power_kw = energy_discharge / dt_hours if dt_hours > 0 else 0
            self.warnings.append(f"放电触达SOC下限 {self.soc_min:.2f}")

        soc_decrease = energy_discharge / self.round_trip_efficiency / self.capacity_kwh
        self.soc -= soc_decrease
        self.total_energy_discharged += energy_discharge

        revenue = energy_discharge * price
        degradation_cost = energy_discharge * self.degradation_cost_per_kwh
        self.total_degradation_cost += degradation_cost
        self.arbitrage_revenue += revenue
        self.arbitrage_revenue -= degradation_cost

        self.soc_history.append(self.soc)
        self.power_history.append(-power_kw)
        self.status_history.append('discharging')

        return {
            'actual_power': power_kw,
            'energy': energy_discharge,
            'revenue': revenue,
            'degradation': degradation_cost,
            'status': 'discharging',
        }

    def idle(self):
        self.soc_history.append(self.soc)
        self.power_history.append(0)
        if self.offline:
            self.status_history.append('offline')
        else:
            self.status_history.append('idle')

    def set_offline(self, offline: bool):
        if offline and not self.offline:
            self.warnings.append("电池进入离线检修状态")
        elif not offline and self.offline:
            self.warnings.append("电池从离线检修恢复")
        self.offline = offline

    def get_available_discharge_energy(self, dt_hours: float) -> float:
        if self.offline:
            return 0.0
        available_soc = self.soc - self.soc_min
        energy_by_soc = available_soc * self.capacity_kwh * self.round_trip_efficiency
        energy_by_power = self.max_power_kw * dt_hours
        return min(energy_by_soc, energy_by_power)

    def get_available_charge_energy(self, dt_hours: float) -> float:
        if self.offline:
            return 0.0
        available_soc = self.soc_max - self.soc
        energy_by_soc = available_soc * self.capacity_kwh / self.round_trip_efficiency
        energy_by_power = self.max_power_kw * dt_hours
        return min(energy_by_soc, energy_by_power)

    def add_dr_revenue(self, amount: float):
        self.dr_revenue += amount

    def add_reg_revenue(self, amount: float):
        self.reg_revenue += amount

    def total_revenue(self) -> float:
        return self.arbitrage_revenue + self.dr_revenue + self.reg_revenue
