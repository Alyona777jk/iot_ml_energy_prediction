import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

def train_prediction_engine(data_path="data/iot_energy_data.csv", model_path="models/rf_energy_model.pkl", plot_path="reports/energy_forecast_plot.png"):
    data_path = os.path.abspath(data_path)
    model_path = os.path.abspath(model_path)
    plot_path = os.path.abspath(plot_path)

    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Target: Прогноз на 24 години вперед
    df['target_24h'] = df['energy_kWh'].shift(-24)

    # Cyclical & Temporal Features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)

    # Lags & Rolling Features
    df['lag_24'] = df['energy_kWh']
    df['lag_48'] = df['energy_kWh'].shift(24)
    df['lag_168'] = df['energy_kWh'].shift(144)
    
    df['rolling_mean_24'] = df['energy_kWh'].rolling(window=24).mean()
    df['rolling_std_24'] = df['energy_kWh'].rolling(window=24).std()

    df['weather_cloudy'] = (df['weather'] == 'cloudy').astype(int)
    df['weather_rainy'] = (df['weather'] == 'rainy').astype(int)

    df.dropna(inplace=True)

    feature_cols = [
        'energy_kWh', 'indoor_temp', 'outdoor_temp', 'illuminance', 'occupancy', 
        'is_weekend', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'lag_24', 'lag_48', 'lag_168', 'rolling_mean_24', 'rolling_std_24',
        'weather_cloudy', 'weather_rainy'
    ]

    X = df[feature_cols]
    y = df['target_24h']
    timestamps = df['timestamp']

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    timestamps_test = timestamps.iloc[split_idx:]

    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mean_actual = y_test.mean()
    rmse_pct = (rmse / mean_actual) * 100

    print("=== Фінальні Результати Оцінки Моделі ===")
    print(f"Середнє споживання: {mean_actual:.2f} kWh")
    print(f"MAE: {mae:.2f} kWh")
    print(f"RMSE: {rmse:.2f} kWh ({rmse_pct:.2f}% від середнього)")
    print(f"R2 Score: {r2:.4f}")

    if rmse_pct < 10 and r2 > 0.85:
        print("[STATUS] Всі метрики успішно відповідають вимогам практичної роботи!")
    else:
        print("[WARNING] Метрики потребують доопрацювання.")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[+] Модель збережено в '{model_path}'")

    # Створення папки та побудова графіка
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plot_points = 168
    plt.figure(figsize=(14, 6))
    plt.plot(timestamps_test.iloc[-plot_points:], y_test.iloc[-plot_points:], label="Фактичне споживання (Actual kWh)", color="#1f77b4", linewidth=2)
    plt.plot(timestamps_test.iloc[-plot_points:], y_pred[-plot_points:], label="Прогноз на 24г (Predicted kWh)", color="#ff7f0e", linestyle="--", linewidth=2)
    plt.title("Smart Building Energy Consumption: Actual vs 24-Hour Forecast", fontsize=14, pad=15)
    plt.xlabel("Дата та час", fontsize=11)
    plt.ylabel("Споживання електроенергії (kWh)", fontsize=11)
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[+] Графік успішно збережено в '{plot_path}'")

if __name__ == "__main__":
    train_prediction_engine()
