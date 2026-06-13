import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Type
from dataclasses import dataclass, field
from .battery import BatteryState


@dataclass
class BacktestResult:
    time_index: pd.DatetimeIndex
    batteries: Dict[str, BatteryState]
    total_arbitrage_revenue: float = 0.0
    total_dr_revenue: float = 0.0
    total_reg_revenue: float = 0.0
    total_degradation_cost: float = 0.0
    net_load_without_bess: List[float] = field(default_factory=list)
    net_load_with_bess: List[float] = field(default_factory=list)
    total_power_history: List[float] = field(default_factory=list)
    dr_participations: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    strategy_name: str = ""

    def total_revenue(self) -> float:
        return (self.total_arbitrage_revenue + self.total_dr_revenue
                + self.total_reg_revenue - self.total_degradation_cost)

    def get_soc_heatmap_data(self) -> pd.DataFrame:
        soc_data = {}
        for bid, bat in self.batteries.items():
            soc_data[bid] = bat.soc_history
        df = pd.DataFrame(soc_data, index=self.time_index)
        return df

    def get_battery_revenue_ranking(self) -> pd.DataFrame:
        rows = []
        for bid, bat in self.batteries.items():
            rows.append({
                'battery_id': bid,
                'arbitrage_revenue': bat.arbitrage_revenue,
                'dr_revenue': bat.dr_revenue,
                'reg_revenue': bat.reg_revenue,
                'degradation_cost': bat.total_degradation_cost,
                'total_revenue': bat.total_revenue(),
                'total_charged': bat.total_energy_charged,
                'total_discharged': bat.total_energy_discharged,
                'offline': bat.offline,
            })
        df = pd.DataFrame(rows).sort_values('total_revenue', ascending=False)
        return df.reset_index(drop=True)


class BacktestEngine:
    def __init__(self, market_data: pd.DataFrame, batteries_df: pd.DataFrame,
                 dr_events: pd.DataFrame, strategy, dt_hours: float = 0.25):
        self.market_data = market_data
        self.batteries_df = batteries_df
        self.dr_events = dr_events
        self.strategy = strategy
        self.dt_hours = dt_hours
        self.time_index = market_data.index
        self.n_steps = len(market_data)

        self.batteries: Dict[str, BatteryState] = {}
        self._init_batteries()

        self.result = BacktestResult(
            time_index=self.time_index,
            batteries=self.batteries,
            strategy_name=strategy.name if hasattr(strategy, 'name') else strategy.__class__.__name__,
        )

    def _init_batteries(self):
        for bid, row in self.batteries_df.iterrows():
            bat = BatteryState(
                battery_id=bid,
                capacity_kwh=row['capacity_kwh'],
                max_power_kw=row['max_power_kw'],
                soc=row['initial_soc'],
                soc_min=row['soc_min'],
                soc_max=row['soc_max'],
                round_trip_efficiency=row['round_trip_efficiency'],
                degradation_cost_per_kwh=row['degradation_cost_per_kwh'],
                offline=False,
            )
            self.batteries[bid] = bat

    def _update_offline_status(self, step_idx: int):
        for bid, row in self.batteries_df.iterrows():
            if row['offline']:
                start = row['offline_start_idx']
                end = start + row['offline_duration']
                is_offline = start <= step_idx < end
                self.batteries[bid].set_offline(is_offline)

    def _get_active_dr_events(self, step_idx: int) -> List[dict]:
        active = []
        if self.dr_events is None or len(self.dr_events) == 0:
            return active
        for _, event in self.dr_events.iterrows():
            if event['start_idx'] <= step_idx <= event['end_idx']:
                active.append(event.to_dict())
        return active

    def _get_upcoming_dr_events(self, step_idx: int, look_ahead: int = 96) -> List[dict]:
        upcoming = []
        if self.dr_events is None or len(self.dr_events) == 0:
            return upcoming
        for _, event in self.dr_events.iterrows():
            if step_idx < event['start_idx'] <= step_idx + look_ahead:
                upcoming.append(event.to_dict())
        return upcoming

    def _calculate_dr_reservation(self, step_idx: int) -> Dict[str, float]:
        reservations = {}
        active_events = self._get_active_dr_events(step_idx)
        upcoming_events = self._get_upcoming_dr_events(step_idx, look_ahead=96)

        all_events = active_events + upcoming_events

        if not all_events:
            return reservations

        total_power_needed = sum(e['power_kw'] for e in all_events)
        online_batteries = [bid for bid, bat in self.batteries.items() if not bat.offline]

        if not online_batteries:
            return reservations

        total_capacity = sum(self.batteries[bid].max_power_kw for bid in online_batteries)

        if total_capacity == 0:
            return reservations

        for bid in online_batteries:
            share = self.batteries[bid].max_power_kw / total_capacity
            reservations[bid] = total_power_needed * share

        return reservations

    def run(self):
        for step_idx in range(self.n_steps):
            self._update_offline_status(step_idx)

            current_data = self.market_data.iloc[step_idx]
            price = current_data['price']
            load = current_data['load']
            pv = current_data['pv']
            reg_up = current_data.get('reg_up_kw', 0)
            reg_down = current_data.get('reg_down_kw', 0)
            reg_price = current_data.get('reg_price', 0)

            dr_reservations = self._calculate_dr_reservation(step_idx)
            active_dr = self._get_active_dr_events(step_idx)

            context = {
                'step_idx': step_idx,
                'time': self.time_index[step_idx],
                'price': price,
                'load': load,
                'pv': pv,
                'reg_up': reg_up,
                'reg_down': reg_down,
                'reg_price': reg_price,
                'dt_hours': self.dt_hours,
                'market_data': self.market_data,
                'dr_reservations': dr_reservations,
                'active_dr_events': active_dr,
            }

            power_commands = self.strategy.decide(self.batteries, context)

            total_power = 0.0

            for bid, bat in self.batteries.items():
                cmd = power_commands.get(bid, 0)

                reserved_power = dr_reservations.get(bid, 0)

                if bat.offline:
                    bat.idle()
                    continue

                if cmd > 0:
                    available_charge = bat.get_available_charge_energy(self.dt_hours) / self.dt_hours
                    actual_charge_power = min(cmd, available_charge)
                    bat.charge(actual_charge_power, self.dt_hours, price)
                    total_power += actual_charge_power

                elif cmd < 0:
                    discharge_power = abs(cmd)
                    available_discharge = bat.get_available_discharge_energy(self.dt_hours) / self.dt_hours

                    if active_dr and reserved_power > 0:
                        reserved_discharge = reserved_power
                        discharge_for_dr = min(discharge_power, reserved_discharge, available_discharge)
                        remaining_discharge = discharge_power - discharge_for_dr

                        if discharge_for_dr > 0:
                            result = bat.discharge(discharge_for_dr, self.dt_hours, price)
                            total_power -= result['actual_power']

                        if remaining_discharge > 0:
                            remaining_available = available_discharge - discharge_for_dr
                            if remaining_available > 0:
                                actual_discharge = min(remaining_discharge, remaining_available)
                                bat.discharge(actual_discharge, self.dt_hours, price)
                                total_power -= actual_discharge
                    else:
                        actual_discharge = min(discharge_power, available_discharge)
                        bat.discharge(actual_discharge, self.dt_hours, price)
                        total_power -= actual_discharge

                else:
                    bat.idle()

            for event in active_dr:
                event_power = event['power_kw']
                compensation = event_power * self.dt_hours * event['compensation_per_kw'] / 4
                self.result.total_dr_revenue += compensation

                online_batteries = [bid for bid, bat in self.batteries.items() if not bat.offline]
                if online_batteries:
                    per_battery = compensation / len(online_batteries)
                    for bid in online_batteries:
                        self.batteries[bid].add_dr_revenue(per_battery)

            total_reg_revenue_step = 0
            if reg_price > 0 and (reg_up > 0 or reg_down > 0):
                total_online_power = sum(bat.max_power_kw for bat in self.batteries.values() if not bat.offline)
                if total_online_power > 0:
                    reg_provision = min(reg_up + reg_down, total_online_power * 0.1)
                    total_reg_revenue_step = reg_provision * self.dt_hours * reg_price
                    self.result.total_reg_revenue += total_reg_revenue_step

                    online_batteries = [bid for bid, bat in self.batteries.items() if not bat.offline]
                    if online_batteries:
                        per_battery = total_reg_revenue_step / len(online_batteries)
                        for bid in online_batteries:
                            self.batteries[bid].add_reg_revenue(per_battery)

            net_load_no_bess = load - pv
            net_load_with_bess = net_load_no_bess + total_power

            self.result.net_load_without_bess.append(net_load_no_bess)
            self.result.net_load_with_bess.append(net_load_with_bess)
            self.result.total_power_history.append(total_power)

        self._finalize_results()
        return self.result

    def _finalize_results(self):
        for bid, bat in self.batteries.items():
            self.result.total_arbitrage_revenue += bat.arbitrage_revenue
            self.result.total_degradation_cost += bat.total_degradation_cost

            if len(bat.warnings) > 0:
                for w in bat.warnings[:5]:
                    self.result.warnings.append(f"[{bid}] {w}")

        dr_participations = []
        if self.dr_events is not None:
            for _, event in self.dr_events.iterrows():
                dr_participations.append({
                    'event_id': event['event_id'],
                    'start_time': event['start_time'],
                    'end_time': event['end_time'],
                    'power_kw': event['power_kw'],
                    'compensation': event.get('total_compensation', 0),
                })
        self.result.dr_participations = dr_participations
