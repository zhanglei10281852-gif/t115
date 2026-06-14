import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_time_index(days=30, freq='15min', start_date=None):
    if start_date is None:
        start_date = datetime(2024, 1, 1)
    periods = int(days * 24 * 60 / int(freq.replace('min', '')))
    return pd.date_range(start=start_date, periods=periods, freq=freq)


def generate_electricity_price(time_index):
    n = len(time_index)
    hour_of_day = time_index.hour.values
    day_of_week = time_index.dayofweek.values

    base_price = 0.5

    daily_pattern = np.zeros(n)
    daily_pattern[(hour_of_day >= 0) & (hour_of_day < 6)] = -0.2
    daily_pattern[(hour_of_day >= 6) & (hour_of_day < 10)] = 0.15
    daily_pattern[(hour_of_day >= 10) & (hour_of_day < 14)] = 0.05
    daily_pattern[(hour_of_day >= 14) & (hour_of_day < 18)] = 0.1
    daily_pattern[(hour_of_day >= 18) & (hour_of_day < 22)] = 0.25
    daily_pattern[(hour_of_day >= 22) & (hour_of_day < 24)] = -0.1

    weekend_factor = np.where(day_of_week >= 5, -0.05, 0.0)

    noise = np.random.normal(0, 0.03, n)

    price = base_price + daily_pattern + weekend_factor + noise
    price = np.maximum(price, 0.2)

    return pd.Series(price, index=time_index, name='price')


def generate_load_curve(time_index):
    n = len(time_index)
    hour_of_day = time_index.hour.values
    day_of_week = time_index.dayofweek.values

    base_load = 5000.0

    daily_pattern = np.zeros(n)
    daily_pattern[(hour_of_day >= 0) & (hour_of_day < 6)] = -0.4
    daily_pattern[(hour_of_day >= 6) & (hour_of_day < 9)] = 0.3
    daily_pattern[(hour_of_day >= 9) & (hour_of_day < 12)] = 0.15
    daily_pattern[(hour_of_day >= 12) & (hour_of_day < 14)] = 0.2
    daily_pattern[(hour_of_day >= 14) & (hour_of_day < 18)] = 0.1
    daily_pattern[(hour_of_day >= 18) & (hour_of_day < 22)] = 0.35
    daily_pattern[(hour_of_day >= 22) & (hour_of_day < 24)] = -0.1

    weekend_factor = np.where(day_of_week >= 5, 0.1, 0.0)

    noise = np.random.normal(0, 0.05, n)

    load = base_load * (1 + daily_pattern + weekend_factor + noise)
    load = np.maximum(load, base_load * 0.3)

    return pd.Series(load, index=time_index, name='load')


def generate_pv_output(time_index):
    n = len(time_index)
    hour_of_day = time_index.hour.values
    day_of_year = time_index.dayofyear.values

    peak_pv = 3000.0

    solar_elevation = np.maximum(
        0,
        np.sin(np.pi * (hour_of_day - 6) / 12)
    )
    solar_elevation[hour_of_day < 6] = 0
    solar_elevation[hour_of_day >= 18] = 0

    seasonal_factor = 0.5 + 0.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

    daily_noise = np.random.normal(1, 0.15, n)
    daily_noise = np.maximum(daily_noise, 0)

    pv = peak_pv * solar_elevation * seasonal_factor * daily_noise
    pv = np.maximum(pv, 0)

    return pd.Series(pv, index=time_index, name='pv')


def generate_battery_swarm(num_batteries=50, offline_ratio=0.1):
    np.random.seed(42)

    battery_ids = [f'BAT_{i:03d}' for i in range(num_batteries)]

    capacities = np.random.normal(10, 3, num_batteries)
    capacities = np.clip(capacities, 5, 20)

    max_power = capacities * np.random.uniform(0.3, 0.7, num_batteries)

    initial_soc = np.random.uniform(0.3, 0.7, num_batteries)

    round_trip_efficiency = np.random.uniform(0.82, 0.95, num_batteries)

    degradation_cost = np.random.uniform(0.02, 0.08, num_batteries)

    offline = np.random.random(num_batteries) < offline_ratio

    offline_start_idx = np.random.randint(0, int(num_batteries * 0.7), num_batteries)
    offline_duration = np.random.randint(10, 50, num_batteries)

    batteries = pd.DataFrame({
        'battery_id': battery_ids,
        'capacity_kwh': capacities,
        'max_power_kw': max_power,
        'initial_soc': initial_soc,
        'round_trip_efficiency': round_trip_efficiency,
        'degradation_cost_per_kwh': degradation_cost,
        'offline': offline,
        'offline_start_idx': offline_start_idx,
        'offline_duration': offline_duration,
        'soc_min': 0.1,
        'soc_max': 0.95,
    })

    batteries = batteries.set_index('battery_id')

    return batteries


def generate_dr_events(time_index, num_events=5):
    n = len(time_index)
    events = []

    for i in range(num_events):
        start_idx = np.random.randint(int(n * 0.1), int(n * 0.9))
        duration_slots = np.random.randint(4, 16)
        power_kw = np.random.uniform(50, 180)
        compensation_per_kw = np.random.uniform(8, 15)

        start_time = time_index[start_idx]
        end_time = time_index[min(start_idx + duration_slots, n - 1)]

        events.append({
            'event_id': f'DR_{i:02d}',
            'start_time': start_time,
            'end_time': end_time,
            'start_idx': start_idx,
            'end_idx': min(start_idx + duration_slots, n - 1),
            'power_kw': power_kw,
            'compensation_per_kw': compensation_per_kw,
            'total_compensation': power_kw * duration_slots / 4 * compensation_per_kw / 4,
        })

    return pd.DataFrame(events)


def generate_regulation_signals(time_index):
    n = len(time_index)
    hour_of_day = time_index.hour.values

    reg_up = np.random.uniform(0, 200, n)
    reg_down = np.random.uniform(0, 200, n)

    reg_up[(hour_of_day >= 18) & (hour_of_day < 22)] *= 1.5
    reg_down[(hour_of_day >= 2) & (hour_of_day < 6)] *= 1.5

    reg_price = np.random.uniform(0.5, 2.0, n)

    return pd.DataFrame({
        'reg_up_kw': reg_up,
        'reg_down_kw': reg_down,
        'reg_price': reg_price,
    }, index=time_index)


def generate_all_data(days=30, num_batteries=50):
    time_index = generate_time_index(days=days)

    price = generate_electricity_price(time_index)
    load = generate_load_curve(time_index)
    pv = generate_pv_output(time_index)
    batteries = generate_battery_swarm(num_batteries=num_batteries)
    dr_events = generate_dr_events(time_index)
    regulation = generate_regulation_signals(time_index)

    market_data = pd.DataFrame({
        'price': price,
        'load': load,
        'pv': pv,
    })
    market_data = pd.concat([market_data, regulation], axis=1)

    return {
        'time_index': time_index,
        'market_data': market_data,
        'batteries': batteries,
        'dr_events': dr_events,
        'regulation': regulation,
    }


if __name__ == '__main__':
    data = generate_all_data(days=3, num_batteries=10)
    print('Market data shape:', data['market_data'].shape)
    print('Batteries count:', len(data['batteries']))
    print('DR events:', len(data['dr_events']))
    print(data['batteries'].head())
    print(data['market_data'].head())
