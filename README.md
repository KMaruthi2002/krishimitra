# KrishiMitra — Backend

FastAPI service powering [KrishiMitra](https://krishimitrakdp.netlify.app/), a weather-driven farming advisor for Indian farmers. Live at **https://krishimitra-s4v4.onrender.com**.

> The React frontend lives in a separate repo: [`krishimitra-frontend`](https://github.com/KMaruthi2002/krishimitra-frontend).

## What it does

- Crop recommendations (XGBoost + DNN ensemble with KB fallback)
- Pest & disease advisory with weather-aware spray windows
- Fertilizer scheduling aligned with rain forecast
- Irrigation planning via Penman–Monteith ET₀
- Multilingual natural-language chat (en, hi, te, kn, ta, mr)
- Live 7–16 day weather for 30+ Indian districts via Open-Meteo (no API key)

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env            # optional: add ANTHROPIC_API_KEY for Claude-polished chat replies
python tests/test_all.py        # 64 tests should pass
python api/main.py              # http://localhost:8000   (docs at /docs)
```

## Project layout

```
.
├── README.md
├── .gitignore
├── .env.example
├── Procfile                    uvicorn api.main:app
├── runtime.txt                 python-3.11.6
├── requirements.txt
├── api/main.py                 FastAPI routes
├── agent/reasoning_engine.py   Perception → reasoning → action
├── data/
│   ├── weather_service.py      Open-Meteo client
│   ├── weather_pipeline.py     LSTM training-data builder
│   └── knowledge_base/agri_kb.py  12 crops, 7 pesticides, 6 fertilizers, 14 weather rules
├── models_pkg/
│   ├── crop/crop_recommender.py    XGBoost + DNN ensemble
│   ├── lstm/attention_lstm.py      Attention-LSTM forecaster (optional)
│   └── nlp/nlp_pipeline.py         6-intent classifier + 8-type NER + NLG
├── scripts/train_all.py        Train models on real Open-Meteo data
├── tests/test_all.py           64 tests
└── configs/settings.py
```

## API surface

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/query` | Natural-language query (main entry) |
| `POST` | `/api/crop/recommend` | Crop recommendation |
| `POST` | `/api/pesticide/advise` | Pest/disease risk + spray calendar |
| `POST` | `/api/fertilizer/schedule` | Fertilizer timing |
| `POST` | `/api/irrigation/plan` | Irrigation deficit schedule |
| `GET`  | `/api/weather/{location}` | Live forecast |
| `GET`  | `/api/knowledge/{crops,pesticides,fertilizers}` | KB lookups |
| `GET`  | `/api/health` | Health probe |

Example:

```bash
curl -X POST https://krishimitra-s4v4.onrender.com/api/query \
  -H "Content-Type: application/json" \
  -d '{"message": "What crop should I plant in Mandya next month?", "language": "en"}'
```

## Deploy (Render)

Connect this repo as a Render web service. Render uses:

- `runtime.txt` → Python 3.11.6
- `requirements.txt` → installed automatically
- `Procfile` → `uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}`

Optional env var: `ANTHROPIC_API_KEY` for Claude-polished chat replies. Without it the agent falls back to the deterministic template NLG so `/api/query` never returns empty.

## Tech stack

Python 3.11 · FastAPI · Uvicorn · XGBoost · NumPy/Pandas/scikit-learn · httpx · Open-Meteo · Optional PyTorch (Attention-LSTM)
