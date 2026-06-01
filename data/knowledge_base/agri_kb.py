"""
Agricultural Knowledge Base
Encodes domain knowledge: crop-weather-soil-pesticide-fertilizer-irrigation relationships.
Acts as the 'knowledge graph' backbone for the agent reasoning engine.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class CropProfile:
    name: str
    ideal_temp_range: Tuple[float, float]  # min, max in Celsius
    ideal_humidity_range: Tuple[float, float]  # min, max %
    ideal_rainfall_range: Tuple[float, float]  # mm per month
    suitable_soils: List[str]
    seasons: List[str]  # kharif, rabi, zaid
    water_requirement_mm_per_day: float
    growth_duration_days: int
    kc_values: List[float]  # crop coefficients per stage [initial, dev, mid, late]
    common_pests: List[str]
    common_diseases: List[str]


@dataclass
class PesticideProfile:
    name: str
    category: str  # fungicide, insecticide, herbicide
    target_pests: List[str]
    target_diseases: List[str]
    dosage_per_litre: str
    dosage_per_hectare: str
    pre_harvest_interval_days: int
    safe_wind_speed_max: float  # km/h
    rain_free_hours_needed: int
    compatible_crops: List[str]
    weather_conditions: Dict[str, any]  # conditions when effective


@dataclass
class FertilizerProfile:
    name: str
    npk_ratio: Tuple[float, float, float]  # N-P-K percentage
    best_application_timing: str  # "pre_rain", "post_rain", "dry"
    absorption_window_hours: int
    suitable_growth_stages: List[str]
    dosage_per_hectare_kg: float
    rain_sensitivity: str  # "high" = don't apply before heavy rain


class AgriculturalKnowledgeBase:
    """Complete agricultural knowledge base with queryable relationships."""

    def __init__(self):
        self.crops = self._build_crop_profiles()
        self.pesticides = self._build_pesticide_profiles()
        self.fertilizers = self._build_fertilizer_profiles()
        self.pest_weather_rules = self._build_pest_weather_rules()
        self.disease_weather_rules = self._build_disease_weather_rules()

    def _build_crop_profiles(self) -> Dict[str, CropProfile]:
        return {
            "Rice": CropProfile(
                name="Rice", ideal_temp_range=(22, 32), ideal_humidity_range=(60, 90),
                ideal_rainfall_range=(100, 200), suitable_soils=["alluvial", "clayey", "loamy"],
                seasons=["kharif"], water_requirement_mm_per_day=6.0, growth_duration_days=120,
                kc_values=[1.05, 1.20, 1.20, 0.90],
                common_pests=["stem_borer", "brown_planthopper", "leaf_folder"],
                common_diseases=["blast", "blight", "sheath_rot"]
            ),
            "Wheat": CropProfile(
                name="Wheat", ideal_temp_range=(10, 25), ideal_humidity_range=(40, 70),
                ideal_rainfall_range=(40, 80), suitable_soils=["alluvial", "loamy", "clayey"],
                seasons=["rabi"], water_requirement_mm_per_day=4.0, growth_duration_days=130,
                kc_values=[0.30, 0.75, 1.15, 0.40],
                common_pests=["aphid", "termite", "armyworm"],
                common_diseases=["rust", "smut", "powdery_mildew"]
            ),
            "Maize": CropProfile(
                name="Maize", ideal_temp_range=(18, 32), ideal_humidity_range=(50, 80),
                ideal_rainfall_range=(60, 120), suitable_soils=["loamy", "sandy", "alluvial"],
                seasons=["kharif", "rabi"], water_requirement_mm_per_day=5.0, growth_duration_days=100,
                kc_values=[0.30, 0.75, 1.20, 0.60],
                common_pests=["stem_borer", "fall_armyworm", "cutworm"],
                common_diseases=["downy_mildew", "turcicum_leaf_blight", "rust"]
            ),
            "Sugarcane": CropProfile(
                name="Sugarcane", ideal_temp_range=(20, 35), ideal_humidity_range=(60, 85),
                ideal_rainfall_range=(100, 200), suitable_soils=["alluvial", "loamy", "black_cotton"],
                seasons=["kharif", "zaid"], water_requirement_mm_per_day=7.0, growth_duration_days=365,
                kc_values=[0.40, 0.75, 1.25, 0.75],
                common_pests=["top_borer", "internode_borer", "whitefly"],
                common_diseases=["red_rot", "smut", "wilt"]
            ),
            "Cotton": CropProfile(
                name="Cotton", ideal_temp_range=(21, 35), ideal_humidity_range=(40, 70),
                ideal_rainfall_range=(60, 120), suitable_soils=["black_cotton", "alluvial", "loamy"],
                seasons=["kharif"], water_requirement_mm_per_day=5.5, growth_duration_days=180,
                kc_values=[0.35, 0.75, 1.15, 0.70],
                common_pests=["bollworm", "whitefly", "jassid", "aphid"],
                common_diseases=["wilt", "root_rot", "grey_mildew"]
            ),
            "Ragi": CropProfile(
                name="Ragi", ideal_temp_range=(20, 30), ideal_humidity_range=(50, 80),
                ideal_rainfall_range=(50, 100), suitable_soils=["red_laterite", "sandy", "loamy"],
                seasons=["kharif"], water_requirement_mm_per_day=3.5, growth_duration_days=110,
                kc_values=[0.30, 0.70, 1.05, 0.50],
                common_pests=["stem_borer", "grasshopper", "aphid"],
                common_diseases=["blast", "smut", "leaf_spot"]
            ),
            "Jowar": CropProfile(
                name="Jowar", ideal_temp_range=(25, 35), ideal_humidity_range=(40, 70),
                ideal_rainfall_range=(40, 80), suitable_soils=["black_cotton", "red_laterite", "loamy"],
                seasons=["kharif", "rabi"], water_requirement_mm_per_day=4.0, growth_duration_days=100,
                kc_values=[0.30, 0.70, 1.10, 0.55],
                common_pests=["shoot_fly", "stem_borer", "head_bug"],
                common_diseases=["grain_mold", "downy_mildew", "anthracnose"]
            ),
            "Groundnut": CropProfile(
                name="Groundnut", ideal_temp_range=(22, 33), ideal_humidity_range=(50, 75),
                ideal_rainfall_range=(50, 100), suitable_soils=["sandy", "loamy", "red_laterite"],
                seasons=["kharif", "rabi"], water_requirement_mm_per_day=4.5, growth_duration_days=110,
                kc_values=[0.30, 0.70, 1.10, 0.60],
                common_pests=["aphid", "jassid", "leaf_miner", "thrips"],
                common_diseases=["tikka_leaf_spot", "rust", "collar_rot"]
            ),
            "Pulses": CropProfile(
                name="Pulses", ideal_temp_range=(15, 30), ideal_humidity_range=(40, 65),
                ideal_rainfall_range=(30, 60), suitable_soils=["loamy", "sandy", "alluvial"],
                seasons=["rabi", "kharif"], water_requirement_mm_per_day=3.0, growth_duration_days=90,
                kc_values=[0.30, 0.65, 1.00, 0.45],
                common_pests=["pod_borer", "aphid", "bruchid"],
                common_diseases=["wilt", "blight", "rust"]
            ),
            "Sunflower": CropProfile(
                name="Sunflower", ideal_temp_range=(20, 30), ideal_humidity_range=(50, 75),
                ideal_rainfall_range=(50, 90), suitable_soils=["loamy", "alluvial", "black_cotton"],
                seasons=["kharif", "rabi"], water_requirement_mm_per_day=5.0, growth_duration_days=95,
                kc_values=[0.30, 0.70, 1.10, 0.55],
                common_pests=["head_borer", "whitefly", "jassid"],
                common_diseases=["downy_mildew", "rust", "alternaria_blight"]
            ),
            "Barley": CropProfile(
                name="Barley", ideal_temp_range=(8, 22), ideal_humidity_range=(35, 60),
                ideal_rainfall_range=(30, 60), suitable_soils=["loamy", "sandy", "saline"],
                seasons=["rabi"], water_requirement_mm_per_day=3.5, growth_duration_days=120,
                kc_values=[0.30, 0.70, 1.10, 0.40],
                common_pests=["aphid", "armyworm"], common_diseases=["rust", "smut", "stripe"]
            ),
            "Millets": CropProfile(
                name="Millets", ideal_temp_range=(25, 35), ideal_humidity_range=(30, 60),
                ideal_rainfall_range=(25, 60), suitable_soils=["sandy", "red_laterite", "loamy"],
                seasons=["kharif"], water_requirement_mm_per_day=2.5, growth_duration_days=80,
                kc_values=[0.25, 0.60, 0.95, 0.40],
                common_pests=["shoot_fly", "stem_borer"], common_diseases=["downy_mildew", "smut"]
            ),
        }

    def _build_pesticide_profiles(self) -> Dict[str, PesticideProfile]:
        return {
            "Mancozeb": PesticideProfile(
                name="Mancozeb", category="fungicide",
                target_pests=[], target_diseases=["blast", "blight", "leaf_spot", "downy_mildew", "tikka_leaf_spot"],
                dosage_per_litre="2.5g/L", dosage_per_hectare="500-750g in 500L water",
                pre_harvest_interval_days=14, safe_wind_speed_max=15.0, rain_free_hours_needed=6,
                compatible_crops=["Rice", "Wheat", "Groundnut", "Maize", "Pulses"],
                weather_conditions={"humidity_min": 70, "temp_range": [18, 32]}
            ),
            "Carbendazim": PesticideProfile(
                name="Carbendazim", category="fungicide",
                target_pests=[], target_diseases=["blast", "sheath_rot", "wilt", "powdery_mildew", "rust"],
                dosage_per_litre="1g/L", dosage_per_hectare="250-500g in 500L water",
                pre_harvest_interval_days=21, safe_wind_speed_max=15.0, rain_free_hours_needed=4,
                compatible_crops=["Rice", "Wheat", "Cotton", "Pulses", "Groundnut"],
                weather_conditions={"humidity_min": 75, "temp_range": [20, 30]}
            ),
            "Imidacloprid": PesticideProfile(
                name="Imidacloprid", category="insecticide_systemic",
                target_pests=["aphid", "whitefly", "jassid", "brown_planthopper", "thrips"],
                target_diseases=[],
                dosage_per_litre="0.3ml/L", dosage_per_hectare="100ml in 500L water",
                pre_harvest_interval_days=14, safe_wind_speed_max=12.0, rain_free_hours_needed=4,
                compatible_crops=["Rice", "Cotton", "Wheat", "Groundnut", "Sunflower"],
                weather_conditions={"humidity_max": 85, "temp_range": [15, 35]}
            ),
            "Chlorpyrifos": PesticideProfile(
                name="Chlorpyrifos", category="insecticide_contact",
                target_pests=["stem_borer", "cutworm", "termite", "bollworm", "pod_borer"],
                target_diseases=[],
                dosage_per_litre="2.5ml/L", dosage_per_hectare="1-1.5L in 500L water",
                pre_harvest_interval_days=21, safe_wind_speed_max=10.0, rain_free_hours_needed=8,
                compatible_crops=["Rice", "Maize", "Cotton", "Sugarcane", "Pulses"],
                weather_conditions={"humidity_max": 80, "temp_range": [15, 35]}
            ),
            "Propiconazole": PesticideProfile(
                name="Propiconazole", category="fungicide",
                target_pests=[], target_diseases=["rust", "smut", "sheath_blight", "grain_mold"],
                dosage_per_litre="1ml/L", dosage_per_hectare="500ml in 500L water",
                pre_harvest_interval_days=28, safe_wind_speed_max=15.0, rain_free_hours_needed=4,
                compatible_crops=["Wheat", "Rice", "Jowar", "Barley"],
                weather_conditions={"humidity_min": 65, "temp_range": [15, 30]}
            ),
            "Thiamethoxam": PesticideProfile(
                name="Thiamethoxam", category="insecticide_systemic",
                target_pests=["aphid", "whitefly", "leaf_miner", "jassid"],
                target_diseases=[],
                dosage_per_litre="0.2g/L", dosage_per_hectare="100g in 500L water",
                pre_harvest_interval_days=14, safe_wind_speed_max=12.0, rain_free_hours_needed=4,
                compatible_crops=["Cotton", "Rice", "Maize", "Groundnut", "Sunflower"],
                weather_conditions={"humidity_max": 85, "temp_range": [15, 35]}
            ),
            "Cypermethrin": PesticideProfile(
                name="Cypermethrin", category="insecticide_contact",
                target_pests=["bollworm", "fall_armyworm", "head_borer", "pod_borer", "shoot_fly"],
                target_diseases=[],
                dosage_per_litre="1ml/L", dosage_per_hectare="500ml in 500L water",
                pre_harvest_interval_days=14, safe_wind_speed_max=10.0, rain_free_hours_needed=6,
                compatible_crops=["Cotton", "Maize", "Pulses", "Sunflower", "Jowar"],
                weather_conditions={"humidity_max": 80, "temp_range": [15, 35]}
            ),
        }

    def _build_fertilizer_profiles(self) -> Dict[str, FertilizerProfile]:
        return {
            "Urea": FertilizerProfile(
                name="Urea", npk_ratio=(46.0, 0.0, 0.0),
                best_application_timing="pre_rain", absorption_window_hours=48,
                suitable_growth_stages=["development", "mid_season"],
                dosage_per_hectare_kg=100.0, rain_sensitivity="high"
            ),
            "DAP": FertilizerProfile(
                name="DAP", npk_ratio=(18.0, 46.0, 0.0),
                best_application_timing="at_sowing", absorption_window_hours=72,
                suitable_growth_stages=["initial", "development"],
                dosage_per_hectare_kg=100.0, rain_sensitivity="medium"
            ),
            "MOP": FertilizerProfile(
                name="MOP", npk_ratio=(0.0, 0.0, 60.0),
                best_application_timing="pre_rain", absorption_window_hours=36,
                suitable_growth_stages=["mid_season", "late_season"],
                dosage_per_hectare_kg=60.0, rain_sensitivity="medium"
            ),
            "NPK_Complex": FertilizerProfile(
                name="NPK Complex (10:26:26)", npk_ratio=(10.0, 26.0, 26.0),
                best_application_timing="at_sowing", absorption_window_hours=48,
                suitable_growth_stages=["initial"],
                dosage_per_hectare_kg=150.0, rain_sensitivity="medium"
            ),
            "Ammonium_Sulphate": FertilizerProfile(
                name="Ammonium Sulphate", npk_ratio=(20.6, 0.0, 0.0),
                best_application_timing="pre_rain", absorption_window_hours=36,
                suitable_growth_stages=["development", "mid_season"],
                dosage_per_hectare_kg=75.0, rain_sensitivity="high"
            ),
            "Zinc_Sulphate": FertilizerProfile(
                name="Zinc Sulphate", npk_ratio=(0.0, 0.0, 0.0),
                best_application_timing="dry", absorption_window_hours=24,
                suitable_growth_stages=["initial", "development"],
                dosage_per_hectare_kg=25.0, rain_sensitivity="low"
            ),
        }

    def _build_pest_weather_rules(self) -> List[Dict]:
        """Rules: weather condition -> pest outbreak probability."""
        return [
            {"pest": "aphid", "condition": "dry_heat", "temp_min": 25, "temp_max": 38,
             "humidity_max": 50, "rainfall_max_14d": 20, "probability_boost": 0.7},
            {"pest": "whitefly", "condition": "hot_dry", "temp_min": 28, "temp_max": 40,
             "humidity_max": 55, "rainfall_max_14d": 15, "probability_boost": 0.65},
            {"pest": "brown_planthopper", "condition": "warm_humid", "temp_min": 25, "temp_max": 32,
             "humidity_min": 80, "rainfall_min_14d": 50, "probability_boost": 0.75},
            {"pest": "stem_borer", "condition": "moderate_humid", "temp_min": 22, "temp_max": 30,
             "humidity_min": 65, "probability_boost": 0.6},
            {"pest": "bollworm", "condition": "warm_moderate", "temp_min": 25, "temp_max": 35,
             "humidity_min": 50, "humidity_max": 75, "probability_boost": 0.55},
            {"pest": "fall_armyworm", "condition": "warm_humid", "temp_min": 22, "temp_max": 32,
             "humidity_min": 70, "rainfall_min_14d": 30, "probability_boost": 0.7},
            {"pest": "thrips", "condition": "dry_warm", "temp_min": 25, "temp_max": 35,
             "humidity_max": 50, "probability_boost": 0.6},
        ]

    def _build_disease_weather_rules(self) -> List[Dict]:
        """Rules: weather condition -> disease outbreak probability."""
        return [
            {"disease": "blast", "condition": "high_humidity_moderate_temp",
             "temp_min": 20, "temp_max": 30, "humidity_min": 80,
             "consecutive_humid_days": 3, "probability_boost": 0.85},
            {"disease": "blight", "condition": "warm_wet",
             "temp_min": 22, "temp_max": 32, "humidity_min": 75,
             "rainfall_min_14d": 60, "probability_boost": 0.75},
            {"disease": "rust", "condition": "cool_humid",
             "temp_min": 15, "temp_max": 25, "humidity_min": 70,
             "probability_boost": 0.7},
            {"disease": "downy_mildew", "condition": "cool_wet",
             "temp_min": 15, "temp_max": 25, "humidity_min": 85,
             "rainfall_min_14d": 40, "probability_boost": 0.8},
            {"disease": "powdery_mildew", "condition": "warm_dry_to_humid",
             "temp_min": 20, "temp_max": 30, "humidity_min": 60, "humidity_max": 80,
             "probability_boost": 0.65},
            {"disease": "wilt", "condition": "hot_waterlogged",
             "temp_min": 28, "temp_max": 38, "humidity_min": 75,
             "rainfall_min_14d": 80, "probability_boost": 0.7},
            {"disease": "smut", "condition": "moderate_humid",
             "temp_min": 25, "temp_max": 32, "humidity_min": 70,
             "probability_boost": 0.55},
        ]

    # ==================== QUERY METHODS ====================

    def get_crop_profile(self, crop_name: str) -> Optional[CropProfile]:
        return self.crops.get(crop_name)

    def get_suitable_crops(self, temp: float, humidity: float, rainfall_monthly: float,
                           soil_type: str, season: str) -> List[Dict]:
        """Rank crops by suitability score for given conditions."""
        results = []
        for name, crop in self.crops.items():
            score = 0.0
            # Temperature fit
            t_min, t_max = crop.ideal_temp_range
            if t_min <= temp <= t_max:
                score += 0.3
            elif abs(temp - t_min) <= 3 or abs(temp - t_max) <= 3:
                score += 0.15
            # Humidity fit
            h_min, h_max = crop.ideal_humidity_range
            if h_min <= humidity <= h_max:
                score += 0.2
            elif abs(humidity - h_min) <= 5 or abs(humidity - h_max) <= 5:
                score += 0.1
            # Rainfall fit
            r_min, r_max = crop.ideal_rainfall_range
            if r_min <= rainfall_monthly <= r_max:
                score += 0.25
            elif abs(rainfall_monthly - r_min) <= 15 or abs(rainfall_monthly - r_max) <= 15:
                score += 0.12
            # Soil fit
            if soil_type in crop.suitable_soils:
                score += 0.15
            # Season fit
            if season in crop.seasons:
                score += 0.10
            if score > 0.3:
                results.append({"crop": name, "score": round(score, 3), "profile": crop})
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def get_pest_risk(self, temp: float, humidity: float, rainfall_14d: float,
                      crop_name: str) -> List[Dict]:
        """Evaluate pest outbreak risk based on weather and crop."""
        crop = self.crops.get(crop_name)
        if not crop:
            return []
        risks = []
        for rule in self.pest_weather_rules:
            if rule["pest"] not in crop.common_pests:
                continue
            match = True
            if "temp_min" in rule and temp < rule["temp_min"]: match = False
            if "temp_max" in rule and temp > rule["temp_max"]: match = False
            if "humidity_min" in rule and humidity < rule["humidity_min"]: match = False
            if "humidity_max" in rule and humidity > rule["humidity_max"]: match = False
            if "rainfall_max_14d" in rule and rainfall_14d > rule["rainfall_max_14d"]: match = False
            if "rainfall_min_14d" in rule and rainfall_14d < rule["rainfall_min_14d"]: match = False
            if match:
                risks.append({
                    "pest": rule["pest"], "probability": rule["probability_boost"],
                    "condition": rule["condition"]
                })
        return sorted(risks, key=lambda x: x["probability"], reverse=True)

    def get_disease_risk(self, temp: float, humidity: float, rainfall_14d: float,
                         crop_name: str) -> List[Dict]:
        """Evaluate disease outbreak risk."""
        crop = self.crops.get(crop_name)
        if not crop:
            return []
        risks = []
        for rule in self.disease_weather_rules:
            if rule["disease"] not in crop.common_diseases:
                continue
            match = True
            if "temp_min" in rule and temp < rule["temp_min"]: match = False
            if "temp_max" in rule and temp > rule["temp_max"]: match = False
            if "humidity_min" in rule and humidity < rule["humidity_min"]: match = False
            if "humidity_max" in rule and humidity > rule.get("humidity_max", 100): match = False
            if "rainfall_min_14d" in rule and rainfall_14d < rule["rainfall_min_14d"]: match = False
            if match:
                risks.append({
                    "disease": rule["disease"], "probability": rule["probability_boost"],
                    "condition": rule["condition"]
                })
        return sorted(risks, key=lambda x: x["probability"], reverse=True)

    def get_pesticides_for_threat(self, threat_name: str, crop_name: str,
                                  wind_speed: float = 0) -> List[Dict]:
        """Find suitable pesticides for a given pest/disease and crop."""
        results = []
        for name, pest in self.pesticides.items():
            is_match = (threat_name in pest.target_pests or threat_name in pest.target_diseases)
            is_compatible = crop_name in pest.compatible_crops
            is_safe_wind = wind_speed <= pest.safe_wind_speed_max
            if is_match and is_compatible:
                results.append({
                    "pesticide": name, "category": pest.category,
                    "dosage": pest.dosage_per_litre, "dosage_hectare": pest.dosage_per_hectare,
                    "pre_harvest_days": pest.pre_harvest_interval_days,
                    "safe_to_spray": is_safe_wind,
                    "rain_free_hours": pest.rain_free_hours_needed,
                    "wind_warning": f"Wind {wind_speed:.1f} km/h exceeds safe limit {pest.safe_wind_speed_max} km/h" if not is_safe_wind else None
                })
        return results

    def get_fertilizer_schedule(self, crop_name: str, growth_stage: str,
                                rainfall_forecast_mm: float,
                                days_to_rain: Optional[int] = None) -> List[Dict]:
        """Recommend fertilizer type and timing based on crop stage and weather."""
        results = []
        for name, fert in self.fertilizers.items():
            if growth_stage not in fert.suitable_growth_stages:
                continue
            timing_ok = True
            timing_note = ""
            if fert.rain_sensitivity == "high" and rainfall_forecast_mm > 50:
                if days_to_rain is not None and days_to_rain < 1:
                    timing_ok = False
                    timing_note = "Delay: heavy rain imminent, risk of nutrient runoff"
                elif days_to_rain is not None and days_to_rain <= 2:
                    timing_note = "Apply now: rain in 1-2 days maximizes absorption"
                else:
                    timing_note = "Apply 24-48 hours before expected rain for optimal uptake"
            elif fert.best_application_timing == "dry" and rainfall_forecast_mm > 30:
                timing_note = "Prefer dry conditions; light rain acceptable"
            else:
                timing_note = "Conditions suitable for application"

            results.append({
                "fertilizer": fert.name, "npk": fert.npk_ratio,
                "dosage_kg_per_ha": fert.dosage_per_hectare_kg,
                "timing_advisory": timing_note, "safe_to_apply": timing_ok,
                "absorption_hours": fert.absorption_window_hours
            })
        return results

    def compute_irrigation_need(self, crop_name: str, growth_stage: str,
                                 et0: float, effective_rainfall_mm: float) -> Dict:
        """Compute irrigation deficit using FAO-56 methodology."""
        crop = self.crops.get(crop_name)
        if not crop:
            return {"error": f"Crop '{crop_name}' not found in knowledge base"}
        stage_idx = {"initial": 0, "development": 1, "mid_season": 2, "late_season": 3}
        idx = stage_idx.get(growth_stage, 1)
        kc = crop.kc_values[idx] if idx < len(crop.kc_values) else 1.0
        crop_water_need = et0 * kc
        deficit = max(0, crop_water_need - effective_rainfall_mm)
        return {
            "crop": crop_name, "growth_stage": growth_stage,
            "kc": kc, "et0_mm_per_day": round(et0, 2),
            "crop_water_need_mm": round(crop_water_need, 2),
            "effective_rainfall_mm": round(effective_rainfall_mm, 2),
            "irrigation_deficit_mm": round(deficit, 2),
            "recommendation": (
                "No irrigation needed, rainfall sufficient" if deficit < 1
                else f"Irrigate {deficit:.1f} mm per day" if deficit < 5
                else f"Critical: irrigate {deficit:.1f} mm per day immediately"
            )
        }


# Singleton
kb = AgriculturalKnowledgeBase()
