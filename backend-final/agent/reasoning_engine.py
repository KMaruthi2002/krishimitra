"""
Agent Reasoning Engine v2 - Uses real weather, produces actionable day-by-day advisory.
"""
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.knowledge_base.agri_kb import AgriculturalKnowledgeBase, kb
from models_pkg.nlp.nlp_pipeline import DialogueManager

class AgentReasoningEngine:
    def __init__(self, knowledge_base=None, crop_recommender=None):
        self.kb = knowledge_base or kb
        self.crop_recommender = crop_recommender
        self.dialogue = DialogueManager()

    def assess_risks(self, summary: Dict, crop_name: str = None) -> Dict:
        temp = summary.get("temp_avg", 25)
        hum = summary.get("humidity_avg", 65)
        rain = summary.get("rainfall_total", 0)
        wind = summary.get("wind_speed_avg", 8)
        temp_min = summary.get("temp_min", 18)
        risks = {}
        if hum > 80 and 20 <= temp <= 30:
            risks["fungal"] = {"level": "HIGH", "score": min(1, (hum-80)/15), "detail": f"Humidity {hum}% + temp {temp}°C = ideal fungal conditions"}
        elif hum > 70:
            risks["fungal"] = {"level": "MODERATE", "score": (hum-70)/20, "detail": f"Humidity {hum}% approaching risk threshold"}
        if rain < 10:
            risks["drought"] = {"level": "HIGH", "score": 0.8, "detail": f"Only {rain}mm rain expected. Ensure irrigation."}
        elif rain < 30:
            risks["drought"] = {"level": "MODERATE", "score": 0.4, "detail": f"Below-average rainfall ({rain}mm)"}
        if rain > 150:
            risks["flood"] = {"level": "HIGH", "score": 0.8, "detail": f"Heavy rainfall ({rain}mm). Check drainage."}
        if temp_min < 5:
            risks["frost"] = {"level": "HIGH", "score": 0.9, "detail": f"Min temp {temp_min}°C. Frost risk."}
        if wind > 15:
            risks["spray_unsafe"] = {"level": "HIGH", "score": 0.9, "detail": f"Wind {wind} km/h too high for spraying"}
        if crop_name:
            risks["pests"] = self.kb.get_pest_risk(temp, hum, rain, crop_name)[:3]
            risks["diseases"] = self.kb.get_disease_risk(temp, hum, rain, crop_name)[:3]
        return risks

    def generate_crop_advisory(self, weather: Dict, soil_type="loamy", season=None) -> Dict:
        summary = weather.get("summary", {})
        temp, hum, rain = summary.get("temp_avg",25), summary.get("humidity_avg",65), summary.get("rainfall_total",100)
        if not season:
            m = datetime.now().month
            season = "kharif" if m in range(6,11) else "rabi" if m in [11,12,1,2,3] else "zaid"
        if self.crop_recommender:
            try:
                recs = self.crop_recommender.predict(temp_avg=temp, humidity_avg=hum, rainfall_monthly=rain, wind_speed_avg=summary.get("wind_speed_avg",8), soil_type=soil_type, season=season)
            except: recs = None
        else: recs = None
        kb_recs = self.kb.get_suitable_crops(temp, hum, rain, soil_type, season)
        if not recs:
            recs = [{"crop": r["crop"], "confidence": r["score"]} for r in kb_recs[:5]]
        for r in recs:
            p = self.kb.get_crop_profile(r["crop"])
            if p:
                r["growth_days"] = p.growth_duration_days
                r["water_need_mm_day"] = p.water_requirement_mm_per_day
                r["seasons"] = p.seasons
                r["suitable_soils"] = p.suitable_soils
        return {"recommendations": recs[:5], "season": season, "weather_summary": summary, "risks": self.assess_risks(summary), "soil_type": soil_type, "location": weather.get("location","")}

    def generate_pesticide_advisory(self, weather: Dict, crop_name: str) -> Dict:
        summary, daily = weather.get("summary",{}), weather.get("daily",[])
        risks = self.assess_risks(summary, crop_name)
        threats = []
        for pr in risks.get("pests",[]): threats.append({"type":"pest","name":pr["pest"],"prob":pr["probability"]})
        for dr in risks.get("diseases",[]): threats.append({"type":"disease","name":dr["disease"],"prob":dr["probability"]})
        all_recs = []
        for t in threats:
            pests = self.kb.get_pesticides_for_threat(t["name"], crop_name, summary.get("wind_speed_avg",8))
            for p in pests: p["for_threat"] = t["name"]; p["threat_probability"] = t["prob"]
            all_recs.extend(pests)
        seen = set(); unique = []
        for r in all_recs:
            if r["pesticide"] not in seen: seen.add(r["pesticide"]); unique.append(r)
        spray_windows = []
        for day in daily[:7]:
            w_ok = day.get("wind_speed",0) < 15; r_ok = day.get("rainfall",0) < 2
            spray_windows.append({"date": day["date"], "day": day.get("day_name",""), "safe": w_ok and r_ok, "wind": day.get("wind_speed",0), "rain": day.get("rainfall",0)})
        return {"crop": crop_name, "risks": risks, "recommendations": unique, "spray_windows": spray_windows, "weather_summary": summary}

    def generate_fertilizer_advisory(self, weather: Dict, crop_name: str, growth_stage="development") -> Dict:
        daily, summary = weather.get("daily",[]), weather.get("summary",{})
        days_to_rain = None
        for i, d in enumerate(daily):
            if d.get("rainfall",0) > 5: days_to_rain = i; break
        schedule = self.kb.get_fertilizer_schedule(crop_name, growth_stage, summary.get("rainfall_total",0), days_to_rain)
        plan = []
        dtr = days_to_rain
        for d in daily[:14]:
            rain = d.get("rainfall",0)
            action = "APPLY N-fertilizer (pre-rain window)" if dtr is not None and 0 < dtr <= 2 and rain < 2 else "SKIP: Heavy rain, runoff risk" if rain > 10 else "Light application OK" if rain < 3 else "Wait for drier conditions"
            plan.append({"date": d["date"], "day": d.get("day_name",""), "rain_mm": rain, "action": action})
            if dtr is not None: dtr -= 1
        return {"crop": crop_name, "growth_stage": growth_stage, "options": schedule, "daily_plan": plan, "weather_summary": summary}

    def generate_irrigation_advisory(self, weather: Dict, crop_name: str, growth_stage="mid_season") -> Dict:
        daily, summary = weather.get("daily",[]), weather.get("summary",{})
        t_avg, t_max, t_min = summary.get("temp_avg",25), summary.get("temp_max",30), summary.get("temp_min",20)
        et0 = max(1.5, min(10, 0.0023 * (t_avg + 17.8) * (max(1, t_max-t_min)**0.5) * 15))
        analysis = self.kb.compute_irrigation_need(crop_name, growth_stage, et0, summary.get("rainfall_daily_avg",0)*0.8)
        sched = []
        for d in daily[:14]:
            rain_eff = d.get("rainfall",0) * 0.8
            deficit = max(0, analysis["crop_water_need_mm"] - rain_eff)
            sched.append({"date": d["date"], "day": d.get("day_name",""), "rain_mm": d.get("rainfall",0), "need_mm": round(analysis["crop_water_need_mm"],1), "irrigate_mm": round(deficit,1)})
        return {"crop": crop_name, "growth_stage": growth_stage, "et0": round(et0,2), "analysis": analysis, "daily_schedule": sched, "weather_summary": summary}

    async def process_query(self, message: str) -> Dict:
        from data.weather_service import get_weather
        nlp = self.dialogue.process_message(message)
        if not nlp["complete"]:
            return {"type": "clarification", "intent": nlp["intent"], "slots": nlp["slots"], "missing": nlp["missing_slots"], "response": nlp.get("clarification","")}
        loc = nlp["slots"].get("LOCATION","Bangalore")
        days = int(nlp["slots"].get("TIMEFRAME","7"))
        crop = nlp["slots"].get("CROP")
        soil = nlp["slots"].get("SOIL_TYPE","loamy")
        stage = nlp["slots"].get("GROWTH_STAGE","development")
        weather = await get_weather(loc, days)
        result = {"type": "advisory", "intent": nlp["intent"], "slots": nlp["slots"], "location": loc, "weather": weather, "advisories": {}}
        intent = nlp["intent"]
        if intent == "crop_recommendation": result["advisories"]["crop"] = self.generate_crop_advisory(weather, soil)
        elif intent == "pesticide_query" and crop: result["advisories"]["pesticide"] = self.generate_pesticide_advisory(weather, crop)
        elif intent == "fertilizer_advice" and crop: result["advisories"]["fertilizer"] = self.generate_fertilizer_advisory(weather, crop, stage)
        elif intent == "irrigation_schedule" and crop: result["advisories"]["irrigation"] = self.generate_irrigation_advisory(weather, crop, stage)
        elif intent == "weather_forecast": result["advisories"]["weather"] = weather
        else: result["advisories"]["general"] = {"message": "I can help with crop recommendations, pesticide advisory, fertilizer scheduling, irrigation planning, and weather forecasts."}

        # Generate smart response using Claude
        result["ai_response"] = await self.generate_smart_response(message, result)
        self.dialogue.reset()
        return result

    async def generate_smart_response(self, user_message: str, advisory_data: Dict) -> str:
        """Use Claude to generate an intelligent, farmer-friendly response."""
        import httpx, os, json

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            return ""

        # Build context from advisory data
        context_parts = []
        weather = advisory_data.get("weather", {})
        summary = weather.get("summary", {})
        location = advisory_data.get("location", "")

        if summary:
            context_parts.append(f"Weather for {location}: Temp {summary.get('temp_min','?')}-{summary.get('temp_max','?')}°C, Humidity {summary.get('humidity_avg','?')}%, Rainfall {summary.get('rainfall_total','?')}mm over {weather.get('forecast_days',7)} days, Wind {summary.get('wind_speed_avg','?')} km/h, {summary.get('rainy_days',0)} rainy days.")

        for mod, data in advisory_data.get("advisories", {}).items():
            if mod == "crop":
                recs = data.get("recommendations", [])[:3]
                crops_str = ", ".join(f"{r['crop']} ({round(r.get('confidence',0)*100)}%)" for r in recs)
                context_parts.append(f"Crop recommendations: {crops_str}")
                risks = data.get("risks", {})
                for k, v in risks.items():
                    if isinstance(v, dict) and v.get("detail"):
                        context_parts.append(f"Risk - {k}: {v['detail']}")
            elif mod == "pesticide":
                risks = data.get("risks", {})
                for p in risks.get("pests", []):
                    context_parts.append(f"Pest risk: {p['pest'].replace('_',' ')} ({round(p['probability']*100)}%)")
                for d in risks.get("diseases", []):
                    context_parts.append(f"Disease risk: {d['disease'].replace('_',' ')} ({round(d['probability']*100)}%)")
                for r in data.get("recommendations", [])[:2]:
                    context_parts.append(f"Recommended pesticide: {r['pesticide']} ({r['category']}) - {r['dosage']}/L for {r.get('for_threat','').replace('_',' ')}")
                safe_days = [w['day'] for w in data.get("spray_windows",[]) if w.get("safe")]
                if safe_days:
                    context_parts.append(f"Safe spray days this week: {', '.join(safe_days)}")
            elif mod == "fertilizer":
                for f in data.get("options", [])[:2]:
                    status = "safe to apply" if f.get("safe_to_apply") else "delay application"
                    context_parts.append(f"Fertilizer: {f['fertilizer']} (NPK {f['npk']}) - {f['dosage_kg_per_ha']} kg/ha - {status}: {f.get('timing_advisory','')}")
            elif mod == "irrigation":
                analysis = data.get("analysis", {})
                context_parts.append(f"Irrigation: Crop needs {analysis.get('crop_water_need_mm','?')}mm/day, deficit {analysis.get('irrigation_deficit_mm','?')}mm/day. {analysis.get('recommendation','')}")

        context = "\n".join(context_parts)

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 300,
                        "system": """You are KrishiMitra (कृषि मित्र), an expert Indian agricultural advisor. You speak to farmers in simple, practical language. Keep responses SHORT (3-5 sentences). Be specific with numbers, dates, and actionable advice. Use the weather and advisory data provided to give precise, location-specific guidance. You can mix Hindi terms naturally (like kharif, rabi, dal, urea). Never say "I don't know" - always give useful farming advice based on the data. Focus on what the farmer should DO this week.""",
                        "messages": [
                            {"role": "user", "content": f"Farmer's question: {user_message}\n\nCurrent data:\n{context}\n\nGive a short, practical response:"}
                        ]
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data["content"][0]["text"]
        except Exception as e:
            print(f"[Claude API] Error: {e}")
        return ""

agent = AgentReasoningEngine(knowledge_base=kb)
