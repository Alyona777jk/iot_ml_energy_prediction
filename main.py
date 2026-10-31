import os
from src.iot_simulator import generate_iot_data
from src.train_model import train_prediction_engine

def run_pipeline():
    print("--- 1. Створення IoT Телеметрії ---")
    generate_iot_data()
    
    print("\n--- 2. Навчання та Інтеграція ML Prediction Engine ---")
    train_prediction_engine()

if __name__ == "__main__":
    run_pipeline()
