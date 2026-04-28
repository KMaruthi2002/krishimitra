"""KrishiMitra - Central Configuration"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
KB_DIR = DATA_DIR / "knowledge_base"

@dataclass
class WeatherConfig:
    api_key: str = os.getenv("OPENWEATHER_API_KEY", "demo_key")
    base_url: str = "https://api.openweathermap.org/data/2.5"
    features: List[str] = field(default_factory=lambda: [
        "temp_min", "temp_max", "humidity", "rainfall", "wind_speed", "pressure"
    ])
    lookback_window: int = 7
    forecast_horizons: List[int] = field(default_factory=lambda: [1, 7, 14, 30])
    train_split: float = 0.70
    val_split: float = 0.15
    districts: Dict[str, tuple] = field(default_factory=lambda: {
        "Mandya": (12.52, 76.90), "Bangalore": (12.97, 77.59),
        "Mysore": (12.30, 76.65), "Dharwad": (15.46, 75.01),
        "Raichur": (16.20, 77.37), "Shimoga": (13.93, 75.57),
        "Mangalore": (12.87, 74.88), "Coimbatore": (11.00, 76.96),
        "Vijayawada": (16.51, 80.65), "Pune": (18.52, 73.86),
    })

@dataclass
class LSTMConfig:
    input_dim: int = 6
    hidden_dim_1: int = 64
    hidden_dim_2: int = 32
    dropout: float = 0.25
    attention_heads: int = 4
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 10
    huber_delta: float = 1.0

@dataclass
class NLPConfig:
    intent_classes: List[str] = field(default_factory=lambda: [
        "crop_recommendation", "pesticide_query", "weather_forecast",
        "irrigation_schedule", "fertilizer_advice", "general_info"
    ])
    entity_types: List[str] = field(default_factory=lambda: [
        "CROP", "LOCATION", "TIMEFRAME", "PEST", "SOIL_TYPE",
        "GROWTH_STAGE", "WEATHER_PARAM", "QUANTITY"
    ])
    confidence_threshold: float = 0.6

@dataclass
class CropConfig:
    crops: List[str] = field(default_factory=lambda: [
        "Rice", "Wheat", "Maize", "Sugarcane", "Cotton", "Jowar",
        "Bajra", "Ragi", "Groundnut", "Sunflower", "Soybean",
        "Pulses", "Barley", "Millets", "Tobacco", "Jute",
        "Coconut", "Arecanut", "Coffee", "Tea", "Rubber", "Pepper"
    ])
    soil_types: List[str] = field(default_factory=lambda: [
        "alluvial", "black_cotton", "red_laterite", "sandy",
        "clayey", "loamy", "saline", "peaty"
    ])
    seasons: List[str] = field(default_factory=lambda: ["kharif", "rabi", "zaid"])
    top_k: int = 3

@dataclass
class PesticideConfig:
    fungal_humidity_threshold: float = 80.0
    fungal_temp_range: tuple = (20.0, 30.0)
    wind_spray_limit: float = 15.0
    rain_spray_buffer_hours: int = 6

@dataclass
class IrrigationConfig:
    growth_stages: List[str] = field(default_factory=lambda: [
        "initial", "development", "mid_season", "late_season"
    ])
    default_kc: Dict[str, List[float]] = field(default_factory=lambda: {
        "Rice": [1.05, 1.20, 1.20, 0.90],
        "Wheat": [0.30, 0.75, 1.15, 0.40],
        "Maize": [0.30, 0.75, 1.20, 0.60],
        "Cotton": [0.35, 0.75, 1.15, 0.70],
        "Sugarcane": [0.40, 0.75, 1.25, 0.75],
    })

weather_cfg = WeatherConfig()
lstm_cfg = LSTMConfig()
nlp_cfg = NLPConfig()
crop_cfg = CropConfig()
pesticide_cfg = PesticideConfig()
irrigation_cfg = IrrigationConfig()
