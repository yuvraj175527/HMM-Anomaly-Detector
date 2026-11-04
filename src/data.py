from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["temperature", "rpm", "vibration"]

def load_training_data():
    return pd.read_csv(ROOT / "dataset" / "normal_sensor_data.csv")[FEATURES]

def load_test_data():
    return pd.read_csv(ROOT / "dataset" / "test_sensor_data.csv")

def fit_scaler(frame):
    scaler = StandardScaler()
    return scaler.fit_transform(frame[FEATURES]), scaler
