# 🌱 KrishiMitra: कृषि मित्र - Your Farming Friend

An intelligent agent that helps Indian farmers make data-driven decisions about **what to plant, when to spray, how to fertilize, and when to water** using real-time weather forecasts and agricultural science.

**No API keys needed.** Uses Open-Meteo (free, unlimited) for live weather data.

## What It Does

| Feature | What the Farmer Gets |
|---------|---------------------|
| 🌾 **Crop Recommendation** | "Plant Ragi (90% match) based on your soil and next month's weather" |
| 🛡️ **Pest & Disease Advisory** | "Blast risk 85% due to high humidity. Spray Mancozeb 2.5g/L. Safe to spray: Wed, Fri" |
| 🧪 **Fertilizer Scheduling** | "Apply Urea tomorrow (rain in 2 days = max absorption). Skip Thursday (heavy rain = runoff)" |
| 💧 **Irrigation Planning** | "Your maize needs 4.2mm/day. Rain covers Monday. Irrigate 4.2mm Tue-Thu" |
| 🌤️ **Weather Forecast** | Real 7-16 day forecast from Open-Meteo for 30+ Indian districts |

## Architecture

```
Farmer (mobile/web/chat)
        │
    ┌───▼────────────┐
    │   NLP Pipeline  │  Intent Classification + Named Entity Recognition
    │   (6 intents,   │  CROP, LOCATION, TIMEFRAME, PEST, SOIL, STAGE
    │    8 entities)   │
    └───┬────────────┘
        │
    ┌───▼────────────┐     ┌─────────────────┐
    │  Open-Meteo    │────▶│  Agent Reasoner  │
    │  Weather API   │     │  Risk Assessment │
    │  (LIVE, free)  │     │  Module Router   │
    └────────────────┘     └───┬─────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ Crop Engine  │  │  Pesticide   │  │  Fert/Irrig  │
    │ XGBoost+DNN  │  │  LightGBM   │  │  Penman-Mon  │
    │  12 crops    │  │  + Rule KB   │  │  + Agri KB   │
    └──────────────┘  └──────────────┘  └──────────────┘
```

## Quick Start (3 commands)

```bash
# 1. Install
pip install numpy pandas scikit-learn xgboost torch fastapi uvicorn httpx pydantic spacy

# 2. Train models (fetches real weather data, takes ~60 seconds)
cd krishimitra && python scripts/train_all.py

# 3. Run
python api/main.py
# Server starts at http://localhost:8000
# API docs at http://localhost:8000/docs
```

Or with Docker:
```bash
docker-compose up --build
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/query` | POST | Natural language query (main endpoint) |
| `/api/crop/recommend` | POST | Crop recommendation |
| `/api/pesticide/advise` | POST | Pest/disease advisory + spray calendar |
| `/api/fertilizer/schedule` | POST | Fertilizer timing with rain alignment |
| `/api/irrigation/plan` | POST | Daily irrigation deficit schedule |
| `/api/weather/{location}` | GET | Live weather forecast (7-16 days) |
| `/api/knowledge/crops` | GET | All crop profiles |
| `/api/knowledge/pesticides` | GET | All pesticide profiles |
| `/api/health` | GET | Service health check |

### Example: Natural Language Query
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"message": "What crop should I plant in Mandya next month?"}'
```

## Testing

```bash
python tests/test_all.py
# 64 tests covering: KB, NLP, Weather, Agent, Risk Assessment
```

## Tech Stack

- **Weather**: Open-Meteo API (free, no key, real data)
- **Deep Learning**: PyTorch (Attention-LSTM for weather forecasting)
- **ML**: XGBoost + DNN ensemble (crop recommendation)
- **NLP**: Custom intent classifier (6 classes) + NER (8 entity types)
- **Knowledge Base**: 12 crops, 7 pesticides, 6 fertilizers, 14 pest/disease rules
- **API**: FastAPI + Uvicorn (async)
- **Frontend**: React + Tailwind (mobile-first farmer dashboard)

## Project Structure

```
krishimitra/
├── api/main.py                    # FastAPI backend
├── agent/reasoning_engine.py      # Core agent (perception-reasoning-action)
├── data/
│   ├── weather_service.py         # Open-Meteo live + historical API
│   ├── weather_pipeline.py        # LSTM training data pipeline
│   └── knowledge_base/agri_kb.py  # Agricultural knowledge (crops/pests/fertilizers)
├── models_pkg/
│   ├── lstm/attention_lstm.py     # Attention-LSTM weather forecaster
│   ├── nlp/nlp_pipeline.py        # Intent + NER + Dialogue + Response Gen
│   └── crop/crop_recommender.py   # XGBoost + DNN ensemble
├── tests/test_all.py              # 64 tests
├── scripts/train_all.py           # Train all models on real data
├── Dockerfile                     # Container deployment
└── docker-compose.yml
```

## Research: Attention-LSTM Performance on Real Data

Trained on 1,095 real weather records (Bangalore, Mandya, Pune) from Open-Meteo:

| Parameter | MAE | RMSE |
|-----------|-----|------|
| Temperature (min) | 1.34°C | 1.61°C |
| Temperature (max) | 1.45°C | 1.84°C |
| Humidity | 7.80% | 9.25% |
| Rainfall | 5.12mm | 8.03mm |
| Wind Speed | 3.00 km/h | 3.82 km/h |
| Pressure | 1.65 hPa | 2.11 hPa |
