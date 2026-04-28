"""KrishiMitra API v2 - Production-ready backend"""
import sys, time, logging, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

from agent.reasoning_engine import AgentReasoningEngine
from data.knowledge_base.agri_kb import kb
from data.weather_service import get_weather, LOCATIONS
from models_pkg.nlp.nlp_pipeline import intent_classifier, ner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("krishimitra")

# Fix numpy serialization for JSON responses
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def sanitize(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    elif isinstance(obj, (np.integer,)): return int(obj)
    elif isinstance(obj, (np.floating,)): return float(obj)
    elif isinstance(obj, (np.bool_,)): return bool(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    return obj

app = FastAPI(title="KrishiMitra", description="Weather-Driven कृषि मित्र - Your Farming Friend", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Load crop recommender if available
crop_rec = None
try:
    from models_pkg.crop.crop_recommender import CropRecommender
    rec = CropRecommender()
    model_path = Path(__file__).parent.parent / "saved_models" / "crop"
    if (model_path / "xgb_model.json").exists():
        rec.load(str(model_path))
        crop_rec = rec
        logger.info("Crop recommender loaded successfully")
except Exception as e:
    logger.warning(f"Crop recommender not loaded: {e}. Using KB fallback.")

agent = AgentReasoningEngine(knowledge_base=kb, crop_recommender=crop_rec)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({elapsed:.2f}s)")
    return response

# Error handler
@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    logger.error(f"Error: {exc}")
    return JSONResponse(status_code=500, content={"error": str(exc), "detail": "Internal server error. Please try again."})

# Models
class QueryReq(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
class CropReq(BaseModel):
    location: str = "Mandya"
    soil_type: str = "loamy"
    season: Optional[str] = None
class PestReq(BaseModel):
    crop: str
    location: str = "Mandya"
class FertReq(BaseModel):
    crop: str
    growth_stage: str = "development"
    location: str = "Mandya"
class IrrigReq(BaseModel):
    crop: str
    growth_stage: str = "mid_season"
    location: str = "Mandya"

# ─── Endpoints ───
@app.get("/")
async def root():
    return {"service": "KrishiMitra", "version": "2.0.0", "status": "running",
            "ml_model": "loaded" if crop_rec else "using_kb_fallback",
            "weather": "open_meteo (free, no key)",
            "crops": len(kb.crops), "pesticides": len(kb.pesticides), "fertilizers": len(kb.fertilizers),
            "locations": sorted(set(k.title() for k in LOCATIONS.keys()))}

@app.get("/api/health")
async def health():
    return {"status": "healthy", "timestamp": time.time(), "model_loaded": crop_rec is not None}

@app.post("/api/query")
async def process_query(req: QueryReq):
    try:
        result = await agent.process_query(req.message)
        return JSONResponse(content=sanitize(result))
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(500, str(e))

@app.post("/api/crop/recommend")
async def crop_rec_endpoint(req: CropReq):
    weather = await get_weather(req.location, 30)
    return JSONResponse(content=sanitize(agent.generate_crop_advisory(weather, req.soil_type, req.season)))

@app.post("/api/pesticide/advise")
async def pest_adv(req: PestReq):
    weather = await get_weather(req.location, 7)
    return JSONResponse(content=sanitize(agent.generate_pesticide_advisory(weather, req.crop)))

@app.post("/api/fertilizer/schedule")
async def fert_sched(req: FertReq):
    weather = await get_weather(req.location, 14)
    return JSONResponse(content=sanitize(agent.generate_fertilizer_advisory(weather, req.crop, req.growth_stage)))

@app.post("/api/irrigation/plan")
async def irrig_plan(req: IrrigReq):
    weather = await get_weather(req.location, 14)
    return JSONResponse(content=sanitize(agent.generate_irrigation_advisory(weather, req.crop, req.growth_stage)))

@app.get("/api/weather/{location}")
async def weather_endpoint(location: str, days: int = 7):
    return await get_weather(location, min(days, 16))

@app.get("/api/knowledge/crops")
async def crops_list():
    return {"crops": [{
        "name": n, "temp": f"{p.ideal_temp_range[0]}-{p.ideal_temp_range[1]}°C",
        "humidity": f"{p.ideal_humidity_range[0]}-{p.ideal_humidity_range[1]}%",
        "rainfall": f"{p.ideal_rainfall_range[0]}-{p.ideal_rainfall_range[1]}mm",
        "soils": p.suitable_soils, "seasons": p.seasons, "growth_days": p.growth_duration_days,
        "water_mm_day": p.water_requirement_mm_per_day,
        "pests": [x.replace("_", " ") for x in p.common_pests],
        "diseases": [x.replace("_", " ") for x in p.common_diseases],
    } for n, p in kb.crops.items()]}

@app.get("/api/knowledge/pesticides")
async def pesticides_list():
    return {"pesticides": [{
        "name": n, "category": p.category.replace("_", " "),
        "targets": [x.replace("_", " ") for x in p.target_pests + p.target_diseases],
        "dosage": p.dosage_per_litre, "crops": p.compatible_crops,
        "pre_harvest_days": p.pre_harvest_interval_days,
        "safe_wind_max": p.safe_wind_speed_max, "rain_free_hours": p.rain_free_hours_needed,
    } for n, p in kb.pesticides.items()]}

@app.get("/api/knowledge/fertilizers")
async def fertilizers_list():
    return {"fertilizers": [{
        "name": n, "npk": f"{f.npk_ratio[0]}-{f.npk_ratio[1]}-{f.npk_ratio[2]}",
        "timing": f.best_application_timing.replace("_", " "), "dose_kg_ha": f.dosage_per_hectare_kg,
        "stages": [s.replace("_", " ") for s in f.suitable_growth_stages],
    } for n, f in kb.fertilizers.items()]}

@app.post("/api/debug/nlp")
async def debug_nlp(req: QueryReq):
    intent = intent_classifier.classify(req.message)
    entities = ner.extract(req.message)
    return {"intent": intent.intent, "confidence": intent.confidence, "scores": intent.all_scores,
            "entities": [{"type": e.type, "value": e.value, "raw": e.raw_text} for e in entities]}

if __name__ == "__main__":
    print("\n🌱 KrishiMitra API starting...")
    print("  Weather: Open-Meteo (free, no API key)")
    print(f"  Crops: {len(kb.crops)} | Pesticides: {len(kb.pesticides)} | Fertilizers: {len(kb.fertilizers)}")
    print(f"  ML Model: {'Loaded' if crop_rec else 'Using KB fallback'}")
    print("  Docs: http://localhost:8000/docs")
    print("  Health: http://localhost:8000/api/health\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
