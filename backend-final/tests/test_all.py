"""
KrishiMitra Test Suite
Tests every module: KB, NLP, weather, crop recommender, agent pipeline.
Run: python tests/test_all.py
"""
import sys, asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

PASS = 0
FAIL = 0

def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {('- ' + detail) if detail else ''}")


def test_knowledge_base():
    print("\n📚 Testing Knowledge Base...")
    from data.knowledge_base.agri_kb import kb

    test("KB has 12+ crops", len(kb.crops) >= 12, f"got {len(kb.crops)}")
    test("KB has 5+ pesticides", len(kb.pesticides) >= 5, f"got {len(kb.pesticides)}")
    test("KB has 4+ fertilizers", len(kb.fertilizers) >= 4, f"got {len(kb.fertilizers)}")
    test("Rice profile exists", kb.get_crop_profile("Rice") is not None)
    test("Rice needs 120 days", kb.get_crop_profile("Rice").growth_duration_days == 120)

    # Crop suitability
    recs = kb.get_suitable_crops(temp=26, humidity=75, rainfall_monthly=150, soil_type="alluvial", season="kharif")
    test("Crop recommendations returned", len(recs) > 0, f"got {len(recs)}")
    top_crops = [r["crop"] for r in recs[:5]]
    test("Rice recommended for kharif+alluvial+high rain", "Rice" in top_crops, f"top: {top_crops}")

    # Pest risk
    pest_risks = kb.get_pest_risk(temp=26, humidity=85, rainfall_14d=80, crop_name="Rice")
    test("Pest risk detected for humid rice", len(pest_risks) > 0)

    # Disease risk
    disease_risks = kb.get_disease_risk(temp=25, humidity=85, rainfall_14d=60, crop_name="Rice")
    test("Disease risk for humid rice", len(disease_risks) > 0)
    disease_names = [d["disease"] for d in disease_risks]
    test("Blast detected in humid conditions", "blast" in disease_names, f"got: {disease_names}")

    # Pesticide lookup
    pests = kb.get_pesticides_for_threat("blast", "Rice", wind_speed=8)
    test("Pesticide found for blast", len(pests) > 0)
    test("Mancozeb or Carbendazim recommended", any(p["pesticide"] in ["Mancozeb", "Carbendazim"] for p in pests))

    # Fertilizer schedule
    ferts = kb.get_fertilizer_schedule("Rice", "development", rainfall_forecast_mm=100, days_to_rain=2)
    test("Fertilizer options returned", len(ferts) > 0)

    # Irrigation
    irrig = kb.compute_irrigation_need("Rice", "mid_season", et0=5.0, effective_rainfall_mm=2.0)
    test("Irrigation deficit computed", irrig["irrigation_deficit_mm"] > 0, f"deficit={irrig['irrigation_deficit_mm']}")
    test("Kc value correct for rice mid-season", irrig["kc"] == 1.2, f"got {irrig['kc']}")


def test_nlp():
    print("\n🗣️ Testing NLP Pipeline...")
    from models_pkg.nlp.nlp_pipeline import IntentClassifier, AgriNER, DialogueManager

    clf = IntentClassifier()

    # Intent classification
    tests = [
        ("What crop should I plant in Mandya?", "crop_recommendation"),
        ("My rice has pest attack", "pesticide_query"),
        ("What's the weather tomorrow?", "weather_forecast"),
        ("How much water does maize need?", "irrigation_schedule"),
        ("When to apply urea?", "fertilizer_advice"),
        ("Hello", "general_info"),
    ]
    for text, expected in tests:
        result = clf.classify(text)
        test(f"Intent '{text[:35]}...' -> {expected}", result.intent == expected, f"got {result.intent}")

    # NER
    ner = AgriNER()
    entities = ner.extract("What crop should I plant in Mandya next month on red soil?")
    types = [e.type for e in entities]
    test("NER extracts LOCATION", "LOCATION" in types)
    test("NER extracts TIMEFRAME", "TIMEFRAME" in types)
    test("NER extracts SOIL_TYPE", "SOIL_TYPE" in types)

    entities2 = ner.extract("My rice has aphid problem in Shimoga at flowering stage")
    types2 = [e.type for e in entities2]
    values2 = [e.value for e in entities2]
    test("NER finds CROP=Rice", "CROP" in types2 and "Rice" in values2)
    test("NER finds PEST=aphid", "PEST" in types2)
    test("NER finds LOCATION=Shimoga", "LOCATION" in types2)
    test("NER finds GROWTH_STAGE", "GROWTH_STAGE" in types2)

    # Dialogue Manager
    dm = DialogueManager()
    r = dm.process_message("What crop for Mandya next month?")
    test("DM detects crop intent", r["intent"] == "crop_recommendation")
    test("DM fills LOCATION slot", "LOCATION" in r["slots"])
    test("DM fills TIMEFRAME slot", "TIMEFRAME" in r["slots"])
    test("DM marks complete", r["complete"])

    dm.reset()
    r2 = dm.process_message("Rice pest problems")
    test("DM asks clarification for missing CROP", r2["intent"] == "pesticide_query")


def test_weather():
    print("\n🌤️ Testing Weather Service...")
    from data.weather_service import get_weather, fetch_historical_sync, _resolve_coords

    # Coordinate resolution
    lat, lon = _resolve_coords("Bangalore")
    test("Bangalore resolved", abs(lat - 12.97) < 0.1 and abs(lon - 77.59) < 0.1)

    lat2, lon2 = _resolve_coords("bengaluru")
    test("Bengaluru = Bangalore", abs(lat2 - lat) < 0.01)

    # Historical data (sync)
    hist = fetch_historical_sync("bangalore", "2025-06-01", "2025-06-30")
    test("Historical data fetched", hist["count"] > 0, f"got {hist['count']} days")
    if hist["records"]:
        r = hist["records"][0]
        test("Has temp_max", "temp_max" in r)
        test("Has rainfall", "rainfall" in r)
        test("Bangalore June temp realistic", 20 < r["temp_max"] < 40, f"got {r['temp_max']}°C")

    # Live forecast (async)
    async def test_live():
        w = await get_weather("Mandya", 7)
        test("Live forecast returned", w is not None)
        test("Has daily data", len(w.get("daily", [])) > 0)
        test("Has summary", "summary" in w)
        test("Source is open_meteo or fallback", w["source"] in ["open_meteo_live", "climate_model_fallback"])
        if w["daily"]:
            d = w["daily"][0]
            test("Daily has temp", "temp_max" in d and "temp_min" in d)
            test("Daily has rainfall", "rainfall" in d)
            test("Temp realistic", 10 < d["temp_max"] < 50, f"got {d['temp_max']}°C")
    asyncio.run(test_live())


def test_agent():
    print("\n🤖 Testing Agent Pipeline...")
    from agent.reasoning_engine import AgentReasoningEngine
    agent = AgentReasoningEngine()

    async def run():
        # Crop advisory
        r = await agent.process_query("What crop should I plant in Mandya next month?")
        test("Crop query returns advisory", r["type"] == "advisory")
        test("Has crop recommendations", "crop" in r.get("advisories", {}))
        crop_data = r["advisories"]["crop"]
        test("Has 3+ recommendations", len(crop_data.get("recommendations", [])) >= 3)
        test("Weather data attached", "weather" in r and r["weather"].get("daily"))
        agent.dialogue.reset()

        # Pesticide advisory
        r2 = await agent.process_query("Rice pest problems in Shimoga")
        test("Pesticide query works", r2["type"] == "advisory")
        test("Has pesticide data", "pesticide" in r2.get("advisories", {}))
        pest_data = r2["advisories"]["pesticide"]
        test("Has spray windows", len(pest_data.get("spray_windows", [])) > 0)
        agent.dialogue.reset()

        # Fertilizer
        r3 = await agent.process_query("When to apply fertilizer to wheat at flowering stage?")
        test("Fertilizer query works", r3["type"] == "advisory")
        fert_data = r3.get("advisories", {}).get("fertilizer", {})
        test("Has daily plan", len(fert_data.get("daily_plan", [])) > 0)
        agent.dialogue.reset()

        # Irrigation
        r4 = await agent.process_query("How much water does maize need?")
        test("Irrigation query works", r4["type"] == "advisory")
        irrig_data = r4.get("advisories", {}).get("irrigation", {})
        test("Has irrigation analysis", "analysis" in irrig_data)
        test("Has daily schedule", len(irrig_data.get("daily_schedule", [])) > 0)
        agent.dialogue.reset()

        # Weather
        r5 = await agent.process_query("Weather forecast for Bangalore next week")
        test("Weather query works", r5["type"] == "advisory")
        test("Has weather data", "weather" in r5.get("advisories", {}))
        agent.dialogue.reset()

        # Clarification
        r6 = await agent.process_query("Tell me about pest problems")
        test("Ambiguous query triggers clarification or advisory", r6["type"] in ["clarification", "advisory"])
        agent.dialogue.reset()

    asyncio.run(run())


def test_risk_assessment():
    print("\n⚠️ Testing Risk Assessment...")
    from agent.reasoning_engine import AgentReasoningEngine
    agent = AgentReasoningEngine()

    # High fungal risk
    risks = agent.assess_risks({"temp_avg": 25, "humidity_avg": 88, "rainfall_total": 80, "wind_speed_avg": 8, "temp_min": 20}, "Rice")
    test("High humidity -> fungal risk HIGH", risks.get("fungal", {}).get("level") == "HIGH")

    # Drought risk
    risks2 = agent.assess_risks({"temp_avg": 32, "humidity_avg": 40, "rainfall_total": 5, "wind_speed_avg": 6, "temp_min": 25})
    test("Low rain -> drought risk HIGH", risks2.get("drought", {}).get("level") == "HIGH")

    # Spray unsafe
    risks3 = agent.assess_risks({"temp_avg": 25, "humidity_avg": 60, "rainfall_total": 50, "wind_speed_avg": 20, "temp_min": 18})
    test("High wind -> spray unsafe", "spray_unsafe" in risks3)


if __name__ == "__main__":
    print("=" * 60)
    print("🌱 KrishiMitra Test Suite")
    print("=" * 60)

    test_knowledge_base()
    test_nlp()
    test_weather()
    test_agent()
    test_risk_assessment()

    print(f"\n{'=' * 60}")
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print(f"{'=' * 60}")
    if FAIL == 0:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️ {FAIL} test(s) need attention")
    sys.exit(0 if FAIL == 0 else 1)
