"""
Weather Data Pipeline
Fetches REAL historical weather data from Open-Meteo (free, no key)
and prepares it for Attention-LSTM training.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import pickle, os

from data.weather_service import fetch_historical_sync, LOCATIONS
from data.knowledge_base.agri_kb import kb


class WeatherPreprocessor:
    """Normalizes weather data and creates LSTM training sequences."""

    FEATURES = ["temp_min", "temp_max", "humidity", "rainfall", "wind_speed", "pressure"]

    def __init__(self, lookback: int = 7):
        self.lookback = lookback
        self.scalers: Dict[str, MinMaxScaler] = {}

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        data = df[self.FEATURES].values.astype(np.float32)
        out = np.zeros_like(data)
        for i, f in enumerate(self.FEATURES):
            s = MinMaxScaler()
            out[:, i] = s.fit_transform(data[:, i:i+1]).ravel()
            self.scalers[f] = s
        return out

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        data = df[self.FEATURES].values.astype(np.float32)
        out = np.zeros_like(data)
        for i, f in enumerate(self.FEATURES):
            out[:, i] = self.scalers[f].transform(data[:, i:i+1]).ravel()
        return out

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        out = np.zeros_like(data)
        for i, f in enumerate(self.FEATURES):
            out[:, i] = self.scalers[f].inverse_transform(data[:, i:i+1]).ravel()
        return out

    def create_sequences(self, data: np.ndarray, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - self.lookback - horizon + 1):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback])
        return np.array(X), np.array(y)

    def save_scalers(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scalers, f)

    def load_scalers(self, path: str):
        with open(path, "rb") as f:
            self.scalers = pickle.load(f)


def fetch_training_data(districts: List[str] = None, years: int = 2) -> pd.DataFrame:
    """Fetch REAL historical weather data from Open-Meteo for multiple districts."""
    if not districts:
        districts = ["bangalore", "mandya", "mysore", "dharwad", "raichur",
                      "pune", "hyderabad", "chennai", "coimbatore", "mangalore"]

    end_date = "2025-12-31"
    start_date = f"{2026 - years}-01-01"

    all_records = []
    for district in districts:
        print(f"  Fetching {district}...")
        result = fetch_historical_sync(district, start_date, end_date)
        if result["records"]:
            for r in result["records"]:
                r["district"] = district
            all_records.extend(result["records"])
            print(f"    Got {len(result['records'])} days")
        else:
            print(f"    Failed: {result.get('error', 'unknown')}")

    df = pd.DataFrame(all_records)
    # Clean: drop rows with any None
    df = df.dropna().reset_index(drop=True)
    # Rename to match our features
    df = df.rename(columns={"temp_max": "temp_max", "temp_min": "temp_min"})
    return df


class CropDataGenerator:
    """Generates crop recommendation training data using KB profiles."""

    def __init__(self, knowledge_base=None):
        self.kb = knowledge_base or kb

    def generate(self, n_samples: int = 8500) -> pd.DataFrame:
        np.random.seed(42)
        crops = list(self.kb.crops.keys())
        soils = ["alluvial", "black_cotton", "red_laterite", "sandy", "clayey", "loamy", "saline", "peaty"]
        seasons = ["kharif", "rabi", "zaid"]
        records = []
        for _ in range(n_samples):
            crop_name = np.random.choice(crops)
            p = self.kb.crops[crop_name]
            t_mid = np.mean(p.ideal_temp_range)
            temp = np.random.normal(t_mid, 5)
            h_mid = np.mean(p.ideal_humidity_range)
            humidity = np.random.normal(h_mid, 12)
            r_mid = np.mean(p.ideal_rainfall_range)
            rainfall = max(0, np.random.normal(r_mid, 30))
            wind = max(0.5, np.random.exponential(5))
            sw = [3.0 if s in p.suitable_soils else 0.5 for s in soils]
            sw = np.array(sw) / sum(sw)
            soil = np.random.choice(soils, p=sw)
            ssw = [3.0 if s in p.seasons else 0.5 for s in seasons]
            ssw = np.array(ssw) / sum(ssw)
            season = np.random.choice(seasons, p=ssw)
            records.append({
                "temp_avg": round(temp, 1), "humidity_avg": round(humidity, 1),
                "rainfall_monthly": round(rainfall, 1), "wind_speed_avg": round(wind, 1),
                "soil_type": soil, "season": season,
                "elevation_m": round(np.random.uniform(50, 1200)), "soil_ph": round(np.random.uniform(5.5, 8.5), 1),
                "crop": crop_name
            })
        return pd.DataFrame(records)
