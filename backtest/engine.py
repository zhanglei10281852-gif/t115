import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Type
from dataclasses import dataclass, field
from .battery import BatteryState


@dataclass
class DRTracking:
    event_id: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    start_idx: int
    end_idx: int
    power_kw_promised: float
    compensation_per_kw: float
    total_energy_delivered: float = 0.0
    total_energy_promised: float = 0.0
    total_compensation_earned: float = 0.0
    full_compensation: float = 0.0
    fulfillment_ratio: float = 0.0
    shortfall_notified: bool = False


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
    dr_trackings: Dict[str, DRTracking] = field(default_factory=dict)
    reg_up_actual_history: List[float] = field(default_factory=list)
    reg_down_actual_history: List[float] = field(default_factory=list)
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

    def get_dr_summary(self) -> pd.DataFrame:
        rows = []
        for eid, tracking in self.dr_trackings.items():
            rows.append({
                'event_id': eid,
                'start_time': tracking.start_time,
                'end_time': tracking.end_time,
                'power_promised_kw': tracking.power_kw_promised,
                'energy_promised_kwh': tracking.total_energy_promised,
                'energy_delivered_kwh': tracking.total_energy_delivered,
                'fulfillment_ratio': tracking.fulfillment_ratio,
                'compensation_earned': tracking.total_compensation_earned,
                'full_compensation': tracking.full_compensation,
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame()


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

        self.dr_trackings: Dict[str, DRTracking] = {}
        self._init_dr_trackings()

        self.result = BacktestResult(
            time_index=self.time_index,
            batteries=self.batteries,
            strategy_name=strategy.name if hasattr(strategy, 'name') else strategy.__class__.__name__,
            dr_trackings=self.dr_trackings,
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

    def _init_dr_trackings(self):
        if self.dr_events is None or len(self.dr_events) == 0:
            return
        for _, event in self.dr_events.iterrows():
            eid = event['event_id']
            duration_slots = event['end_idx'] - event['start_idx'] + 1
            total_energy_promised = event['power_kw'] * self.dt_hours * duration_slots
            full_comp = event['power_kw'] * self.dt_hours * event['compensation_per_kw'] * duration_slots

            self.dr_trackings[eid] = DRTracking(
                event_id=eid,
                start_time=event['start_time'],
                end_time=event['end_time'],
                start_idx=event['start_idx'],
                end_idx=event['end_idx'],
                power_kw_promised=event['power_kw'],
                compensation_per_kw=event['compensation_per_kw'],
                total_energy_promised=total_energy_promised,
                full_compensation=full_comp,
            )

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

    def _allocate_dr_power(self, event_power_kw: float) -> Dict[str, float]:
        online_batteries = [(bid, bat) for bid, bat in self.batteries.items() if not bat.offline]
        if not online_batteries:
            return {}

        total_available = sum(
            min(bat.max_power_kw, bat.get_available_discharge_energy(self.dt_hours) / self.dt_hours)
            for _, bat in online_batteries
        )

        if total_available == 0:
            return {}

        allocations = {}
        remaining_power = event_power_kw

        for bid, bat in online_batteries:
            bat_available = min(bat.max_power_kw, bat.get_available_discharge_energy(self.dt_hours) / self.dt_hours)
            share = bat_available / total_available
            allocated = min(remaining_power, bat_available, event_power_kw * share)
            if allocated > 0:
                allocations[bid] = allocated
                remaining_power -= allocated

        return allocations

    def run(self):
        for step_idx in range(self.n_steps):
            self._update_offline_status(step_idx)

            current_data = self.market_data.iloc[step_idx]
            price = current_data['price']
            load = current_data['load']
            pv = current_data['pv']
            reg_up_demand = current_data.get('reg_up_kw', 0)
            reg_down_demand = current_data.get('reg_down_kw', 0)
            reg_price = current_data.get('reg_price', 0)

            active_dr = self._get_active_dr_events(step_idx)
            dr_reservations = self._calculate_dr_reservation(step_idx)

            context = {
                'step_idx': step_idx,
                'time': self.time_index[step_idx],
                'price': price,
                'load': load,
                'pv': pv,
                'reg_up': reg_up_demand,
                'reg_down': reg_down_demand,
                'reg_price': reg_price,
                'dt_hours': self.dt_hours,
                'market_data': self.market_data,
                'dr_reservations': dr_reservations,
                'active_dr_events': active_dr,
            }

            power_commands = self.strategy.decide(self.batteries, context)

            total_power = 0.0

            dr_total_discharge_this_step = 0.0
            dr_allocations_this_step: Dict[str, float] = {}

            for event in active_dr:
                alloc = self._allocate_dr_power(event['power_kw'])
                for bid, power in alloc.items():
                    dr_allocations_this_step[bid] = dr_allocations_this_step.get(bid, 0) + power

            for bid, bat in self.batteries.items():
                if bat.offline:
                    bat.idle()
                    continue

                cmd = power_commands.get(bid, 0)
                dr_power = dr_allocations_this_step.get(bid, 0)

                if cmd > 0:
                    available_charge = bat.get_available_charge_energy(self.dt_hours) / self.dt_hours
                    actual_charge_power = min(cmd, available_charge)
                    bat.charge(actual_charge_power, self.dt_hours, price)
                    total_power += actual_charge_power

                elif cmd < 0:
                    discharge_requested = abs(cmd)
                    available_discharge = bat.get_available_discharge_energy(self.dt_hours) / self.dt_hours

                    if dr_power > 0:
                        dr_actual = min(dr_power, available_discharge)
                        if dr_actual > 0:
                            result = bat.discharge(dr_actual, self.dt_hours, price, is_dr_discharge=True)
                            total_power -= result['actual_power']
                            dr_total_discharge_this_step += result['energy']
                            available_discharge -= dr_actual

                        arbitrage_discharge = min(discharge_requested, available_discharge)
                        if arbitrage_discharge > 0:
                            result = bat.discharge(arbitrage_discharge, self.dt_hours, price)
                            total_power -= result['actual_power']
                    else:
                        actual_discharge = min(discharge_requested, available_discharge)
                        result = bat.discharge(actual_discharge, self.dt_hours, price)
                        total_power -= result['actual_power']

                else:
                    if dr_power > 0:
                        available_discharge = bat.get_available_discharge_energy(self.dt_hours) / self.dt_hours
                        dr_actual = min(dr_power, available_discharge)
                        if dr_actual > 0:
                            result = bat.discharge(dr_actual, self.dt_hours, price, is_dr_discharge=True)
                            total_power -= result['actual_power']
                            dr_total_discharge_this_step += result['energy']
                    else:
                        bat.idle()

            for event in active_dr:
                eid = event['event_id']
                promised_energy = event['power_kw'] * self.dt_hours
                delivered = dr_total_discharge_this_step

                if eid in self.dr_trackings:
                    tracking = self.dr_trackings[eid]
                    actual_delivered = min(delivered, promised_energy)
                    tracking.total_energy_delivered += actual_delivered

                    fulfillment = actual_delivered / promised_energy if promised_energy > 0 else 0
                    step_compensation = event['power_kw'] * fulfillment * self.dt_hours * event['compensation_per_kw']
                    tracking.total_compensation_earned += step_compensation

                    self.result.total_dr_revenue += step_compensation

                    if actual_delivered < promised_energy and not tracking.shortfall_notified:
                        shortfall_pct = (1 - fulfillment) * 100
                        self.result.warnings.append(
                            f"[DR] {eid} 时段 {step_idx}: 承诺 {promised_energy:.1f}kWh, "
                            f"实际 {actual_delivered:.1f}kWh, 兑现率 {fulfillment:.1%}, "
                            f"缺口 {shortfall_pct:.1f}%"
                        )
                        tracking.shortfall_notified = True

            actual_reg_up = 0.0
            actual_reg_down = 0.0
            if reg_price > 0 and (reg_up_demand > 0 or reg_down_demand > 0):
                for bid, bat in self.batteries.items():
                    if bat.offline:
                        continue

                    if reg_up_demand > 0:
                        reg_up_cap = bat.get_reg_up_capacity()
                        actual_reg_up += reg_up_cap
                        bat.reg_up_capacity_provided += reg_up_cap

                    if reg_down_demand > 0:
                        reg_down_cap = bat.get_reg_down_capacity()
                        actual_reg_down += reg_down_cap
                        bat.reg_down_capacity_provided += reg_down_cap

                reg_up_provision = min(actual_reg_up, reg_up_demand) * 0.3
                reg_down_provision = min(actual_reg_down, reg_down_demand) * 0.3

                reg_up_revenue = reg_up_provision * self.dt_hours * reg_price
                reg_down_revenue = reg_down_provision * self.dt_hours * reg_price * 0.5
                total_reg_step = reg_up_revenue + reg_down_revenue

                self.result.total_reg_revenue += total_reg_step
                self.result.reg_up_actual_history.append(reg_up_provision)
                self.result.reg_down_actual_history.append(reg_down_provision)

                online_batteries = [bid for bid, bat in self.batteries.items() if not bat.offline]
                if online_batteries and total_reg_step > 0:
                    total_cap = sum(
                        self.batteries[bid].reg_up_capacity_provided +
                        self.batteries[bid].reg_down_capacity_provided
                        for bid in online_batteries
                    )
                    if total_cap > 0:
                        for bid in online_batteries:
                            bat = self.batteries[bid]
                            share = (bat.reg_up_capacity_provided + bat.reg_down_capacity_provided) / total_cap
                            bat.add_reg_revenue(total_reg_step * share)
            else:
                self.result.reg_up_actual_history.append(0)
                self.result.reg_down_actual_history.append(0)

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

        for eid, tracking in self.dr_trackings.items():
            if tracking.total_energy_promised > 0:
                tracking.fulfillment_ratio = tracking.total_energy_delivered / tracking.total_energy_promised
            else:
                tracking.fulfillment_ratio = 0.0

            self.result.dr_participations.append({
                'event_id': eid,
                'start_time': tracking.start_time,
                'end_time': tracking.end_time,
                'power_kw_promised': tracking.power_kw_promised,
                'energy_promised_kwh': tracking.total_energy_promised,
                'energy_delivered_kwh': tracking.total_energy_delivered,
                'fulfillment_ratio': tracking.fulfillment_ratio,
                'compensation_earned': tracking.total_compensation_earned,
                'full_compensation_possible': tracking.full_compensation,
            })
