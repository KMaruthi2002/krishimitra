"""
Weather Service - Open-Meteo API (completely free, no API key)
Provides:
  1. Live 7-16 day weather forecasts for any location
  2. Historical weather data (2000-present) for LSTM training
  3. Location geocoding for Indian districts
"""
import httpx
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORY_URL = "https://archive-api.open-meteo.com/v1/archive"

# 30+ Indian agricultural districts with coordinates
LOCATIONS: Dict[str, Tuple[float, float]] = {
    "mandya": (12.52, 76.90), "bangalore": (12.97, 77.59), "bengaluru": (12.97, 77.59),
    "mysore": (12.30, 76.65), "mysuru": (12.30, 76.65), "dharwad": (15.46, 75.01),
    "raichur": (16.20, 77.37), "shimoga": (13.93, 75.57), "shivamogga": (13.93, 75.57),
    "mangalore": (12.87, 74.88), "mangaluru": (12.87, 74.88),
    "coimbatore": (11.00, 76.96), "vijayawada": (16.51, 80.65),
    "pune": (18.52, 73.86), "hyderabad": (17.38, 78.49), "chennai": (13.08, 80.27),
    "mumbai": (19.07, 72.87), "kolkata": (22.57, 88.36), "delhi": (28.61, 77.21),
    "lucknow": (26.85, 80.95), "jaipur": (26.92, 75.78), "bhopal": (23.26, 77.41),
    "patna": (25.61, 85.14), "ranchi": (23.34, 85.31), "guwahati": (26.14, 91.74),
    "thiruvananthapuram": (8.52, 76.94), "kochi": (9.93, 76.27),
    "madurai": (9.92, 78.12), "warangal": (17.98, 79.60), "hassan": (13.00, 76.10),
    "davangere": (14.47, 75.92), "hubli": (15.36, 75.12), "belgaum": (15.85, 74.50),
    "belagavi": (15.85, 74.50), "gulbarga": (17.33, 76.83), "kalaburagi": (17.33, 76.83),
    "nagpur": (21.15, 79.09), "indore": (22.72, 75.86), "chandigarh": (30.73, 76.78),
    "dehradun": (30.32, 78.03), "varanasi": (25.32, 83.01),
}

DAILY_PARAMS = "temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,precipitation_sum,wind_speed_10m_max,surface_pressure_mean"


def _resolve_coords(location: str) -> Tuple[float, float]:
    """Resolve location name to coordinates."""
    loc = location.lower().strip()
    if loc in LOCATIONS:
        return LOCATIONS[loc]
    # Fuzzy match
    for key, coords in LOCATIONS.items():
        if loc in key or key in loc:
            return coords
    return LOCATIONS.get("bangalore", (12.97, 77.59))


async def get_live_forecast(location: str, days: int = 7) -> Dict:
    """
    Fetch live weather forecast from Open-Meteo.
    No API key required. Returns up to 16 days of forecast.
    """
    lat, lon = _resolve_coords(location)
    days = min(days, 16)  # Open-Meteo max is 16 days

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(FORECAST_URL, params={
                "latitude": lat, "longitude": lon,
                "daily": DAILY_PARAMS,
                "timezone": "Asia/Kolkata",
                "forecast_days": days,
            })
            resp.raise_for_status()
            data = resp.json()

        daily_data = data["daily"]
        daily_forecasts = []
        for i in range(len(daily_data["time"])):
            t_max = daily_data["temperature_2m_max"][i] or 30
            t_min = daily_data["temperature_2m_min"][i] or 20
            hum = daily_data["relative_humidity_2m_mean"][i] or 60
            rain = daily_data["precipitation_sum"][i] or 0
            wind = daily_data["wind_speed_10m_max"][i] or 5
            pressure = daily_data.get("surface_pressure_mean", [None]*16)[i] or 1013

            date_str = daily_data["time"][i]
            dt = datetime.strptime(date_str, "%Y-%m-%d")

            daily_forecasts.append({
                "date": date_str,
                "day_name": dt.strftime("%A"),
                "temp_min": round(t_min, 1),
                "temp_max": round(t_max, 1),
                "temp_avg": round((t_min + t_max) / 2, 1),
                "humidity": round(hum, 1),
                "rainfall": round(rain, 1),
                "wind_speed": round(wind, 1),
                "pressure": round(pressure, 1),
                "description": (
                    "heavy rain" if rain > 20 else
                    "moderate rain" if rain > 5 else
                    "light rain" if rain > 0.5 else
                    "partly cloudy" if hum > 70 else "clear sky"
                ),
            })

        # Compute summary
        temps_avg = [d["temp_avg"] for d in daily_forecasts]
        rains = [d["rainfall"] for d in daily_forecasts]
        hums = [d["humidity"] for d in daily_forecasts]
        winds = [d["wind_speed"] for d in daily_forecasts]

        return {
            "location": location.title(),
            "coordinates": {"lat": lat, "lon": lon},
            "generated_at": datetime.now().isoformat(),
            "forecast_days": len(daily_forecasts),
            "source": "open_meteo_live",
            "daily": daily_forecasts,
            "summary": {
                "temp_min": round(min(d["temp_min"] for d in daily_forecasts), 1),
                "temp_max": round(max(d["temp_max"] for d in daily_forecasts), 1),
                "temp_avg": round(np.mean(temps_avg), 1),
                "humidity_avg": round(np.mean(hums), 1),
                "rainfall_total": round(sum(rains), 1),
                "rainfall_daily_avg": round(np.mean(rains), 1),
                "wind_speed_avg": round(np.mean(winds), 1),
                "rainy_days": sum(1 for r in rains if r > 0.5),
            }
        }

    except Exception as e:
        print(f"[WeatherService] Live forecast failed: {e}. Using climate model fallback.")
        return _generate_fallback(location, days)


def fetch_historical_sync(location: str, start_date: str, end_date: str) -> Dict:
    """
    Fetch historical weather data (synchronous, for training).
    Returns daily weather records from Open-Meteo archive.
    """
    lat, lon = _resolve_coords(location)
    try:
        resp = httpx.get(HISTORY_URL, params={
            "latitude": lat, "longitude": lon,
            "start_date": start_date, "end_date": end_date,
            "daily": "temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,precipitation_sum,wind_speed_10m_max,surface_pressure_mean",
            "timezone": "Asia/Kolkata",
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        daily = data["daily"]

        records = []
        for i in range(len(daily["time"])):
            records.append({
                "date": daily["time"][i],
                "temp_max": daily["temperature_2m_max"][i] or 30,
                "temp_min": daily["temperature_2m_min"][i] or 20,
                "humidity": daily["relative_humidity_2m_mean"][i] or 60,
                "rainfall": daily["precipitation_sum"][i] or 0,
                "wind_speed": daily["wind_speed_10m_max"][i] or 5,
                "pressure": (daily.get("surface_pressure_mean") or [1013]*len(daily["time"]))[i] or 1013,
            })

        return {
            "location": location,
            "coordinates": {"lat": lat, "lon": lon},
            "start_date": start_date,
            "end_date": end_date,
            "records": records,
            "count": len(records),
            "source": "open_meteo_archive",
        }
    except Exception as e:
        print(f"[WeatherService] Historical fetch failed for {location}: {e}")
        return {"location": location, "records": [], "count": 0, "error": str(e)}


def _generate_fallback(location: str, days: int) -> Dict:
    """Climate model fallback when API is unreachable."""
    now = datetime.now()
    month = now.month
    loc = location.lower()

    # Regional base temps
    base_temps = {"mangalore": 28, "chennai": 29, "delhi": 25, "kolkata": 27,
                  "mumbai": 28, "bangalore": 24, "hyderabad": 28, "pune": 26}
    base = base_temps.get(loc, 26)
    is_monsoon = month in [6, 7, 8, 9]

    np.random.seed(hash(loc + str(now.date())) % 2**31)
    daily = []
    for d in range(days):
        date = now + timedelta(days=d)
        t_avg = base + np.random.normal(0, 2)
        rain = max(0, np.random.exponential(15 if is_monsoon else 2)) if np.random.random() < (0.5 if is_monsoon else 0.1) else 0
        daily.append({
            "date": date.strftime("%Y-%m-%d"), "day_name": date.strftime("%A"),
            "temp_min": round(t_avg - 4, 1), "temp_max": round(t_avg + 4, 1),
            "temp_avg": round(t_avg, 1), "humidity": round(60 + np.random.normal(0, 10), 1),
            "rainfall": round(rain, 1), "wind_speed": round(max(1, np.random.exponential(5)), 1),
            "pressure": round(1013 + np.random.normal(0, 3), 1),
            "description": "rain" if rain > 1 else "clear sky",
        })

    return {
        "location": location.title(), "coordinates": _resolve_coords(location),
        "generated_at": now.isoformat(), "forecast_days": days,
        "source": "climate_model_fallback", "daily": daily,
        "summary": {
            "temp_min": min(d["temp_min"] for d in daily),
            "temp_max": max(d["temp_max"] for d in daily),
            "temp_avg": round(np.mean([d["temp_avg"] for d in daily]), 1),
            "humidity_avg": round(np.mean([d["humidity"] for d in daily]), 1),
            "rainfall_total": round(sum(d["rainfall"] for d in daily), 1),
            "rainfall_daily_avg": round(np.mean([d["rainfall"] for d in daily]), 1),
            "wind_speed_avg": round(np.mean([d["wind_speed"] for d in daily]), 1),
            "rainy_days": sum(1 for d in daily if d["rainfall"] > 0.5),
        }
    }


# Main entry point
async def get_weather(location: str, days: int = 7) -> Dict:
    """Get weather forecast. Uses Open-Meteo live API (no key needed)."""
    return await get_live_forecast(location, days)
