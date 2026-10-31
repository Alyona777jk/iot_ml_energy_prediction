import pandas as pd
import numpy as np
import os

def generate_iot_data(days=90, save_path="data/iot_energy_data.csv"):
    np.random.seed(42)
    timestamps = pd.date_range(start="2025-01-01", periods=days*24, freq="h")
    n_samples = len(timestamps)

    hour = timestamps.hour
    dayofweek = timestamps.dayofweek
    is_weekend = (dayofweek >= 5).astype(int)

    # Добові та тижневі патерни
    daily_cycle = np.sin(2 * np.pi * (hour - 7) / 24)
    outdoor_temp = 12 + 8 * daily_cycle + np.random.normal(0, 0.5, n_samples)
    indoor_temp = 21 + 2 * daily_cycle + np.random.normal(0, 0.2, n_samples)
    
    # Залежність відвідуваності від часу
    base_occupancy = np.where((hour >= 8) & (hour <= 18) & (is_weekend == 0), 100, 5)
    occupancy = base_occupancy + np.random.normal(0, 2, n_samples)
    occupancy = np.maximum(0, occupancy).astype(int)

    illuminance = np.maximum(0, 800 * np.sin(np.pi * (hour - 6) / 12)) * (1 - is_weekend * 0.5)

    weather_cycle = (np.sin(2 * np.pi * np.arange(n_samples) / (24 * 3)) + 1) / 2
    weather_conditions = np.where(weather_cycle > 0.6, "sunny", np.where(weather_cycle > 0.3, "cloudy", "rainy"))

    # Формула споживання електроенергії
    base_consumption = 30.0
    occupancy_effect = occupancy * 0.4
    temp_effect = np.abs(indoor_temp - outdoor_temp) * 1.5
    light_effect = (illuminance > 50).astype(int) * 5.0
    noise = np.random.normal(0, 0.8, n_samples)

    energy_kWh = base_consumption + occupancy_effect + temp_effect + light_effect + noise
    energy_kWh = np.maximum(5.0, energy_kWh)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "energy_kWh": np.round(energy_kWh, 2),
        "indoor_temp": np.round(indoor_temp, 2),
        "outdoor_temp": np.round(outdoor_temp, 2),
        "illuminance": np.round(illuminance, 2),
        "occupancy": occupancy,
        "day_of_week": dayofweek,
        "hour": hour,
        "weather": weather_conditions
    })

    abs_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    df.to_csv(abs_path, index=False)
    print(f"[+] Дані збережено в '{abs_path}'. Записів: {len(df)}")

if __name__ == "__main__":
    generate_iot_data()
