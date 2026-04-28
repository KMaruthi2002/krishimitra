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
        self.dialogue.reset()
        return result

agent = AgentReasoningEngine(knowledge_base=kb)
