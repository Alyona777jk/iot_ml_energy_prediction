import pandas as pd
import numpy as np
import joblib

def run_realtime_inference():
    model_path = "models/rf_energy_model.pkl"
    data_path = "data/iot_energy_data.csv"
    
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)

    df['lag_24'] = df['energy_kWh']
    df['lag_48'] = df['energy_kWh'].shift(24)
    df['lag_168'] = df['energy_kWh'].shift(144)
    
    df['rolling_mean_24'] = df['energy_kWh'].rolling(window=24).mean()
    df['rolling_std_24'] = df['energy_kWh'].rolling(window=24).std()

    df['weather_cloudy'] = (df['weather'] == 'cloudy').astype(int)
    df['weather_rainy'] = (df['weather'] == 'rainy').astype(int)

    # Точний збіг колонок із train_model.py
    feature_cols = [
        'energy_kWh', 'indoor_temp', 'outdoor_temp', 'illuminance', 'occupancy', 
        'is_weekend', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'lag_24', 'lag_48', 'lag_168', 'rolling_mean_24', 'rolling_std_24',
        'weather_cloudy', 'weather_rainy'
    ]

    latest_telemetry = df[feature_cols].iloc[[-1]]
    latest_time = df['timestamp'].iloc[-1]

    predicted_energy = model.predict(latest_telemetry)[0]

    print("=== IoT Real-Time Inference Check ===")
    print(f"Поточний час телеметрії:  {latest_time}")
    print(f"Поточне споживання:       {df['energy_kWh'].iloc[-1]:.2f} kWh")
    print(f"Температура в приміщенні:  {latest_telemetry['indoor_temp'].values[0]} °C")
    print(f"Людей у будівлі:          {latest_telemetry['occupancy'].values[0]}")
    print("------------------------------------")
    print(f"[PROGNOSIS] Прогнозоване споживання через 24 години: {predicted_energy:.2f} kWh")

if __name__ == "__main__":
    run_realtime_inference()
