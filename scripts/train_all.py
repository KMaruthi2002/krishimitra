"""
KrishiMitra Training Script
Trains LSTM on REAL historical weather data from Open-Meteo (free, no key).
Trains crop recommender on synthetic data grounded in agricultural KB.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def train_weather_model():
    print("=" * 60)
    print("TRAINING: Attention-LSTM on REAL Weather Data")
    print("=" * 60)
    from data.weather_pipeline import fetch_training_data, WeatherPreprocessor
    from models_pkg.lstm.attention_lstm import WeatherForecaster

    print("\n[1/3] Fetching real historical weather from Open-Meteo (free API)...")
    df = fetch_training_data(
        districts=["bangalore", "mandya", "pune", "hyderabad", "chennai"],
        years=2
    )
    print(f"  Total: {len(df)} real weather records across {df['district'].nunique()} districts")

    print("\n[2/3] Preprocessing sequences...")
    pp = WeatherPreprocessor(lookback=7)
    all_X, all_y = [], []
    for district in df["district"].unique():
        ddf = df[df["district"] == district].sort_values("date").reset_index(drop=True)
        if len(ddf) < 30:
            continue
        norm = pp.fit_transform(ddf)
        X, y = pp.create_sequences(norm, horizon=1)
        all_X.append(X)
        all_y.append(y)
    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    n = len(X)
    t1, t2 = int(n * 0.7), int(n * 0.85)
    print(f"  Train: {t1} | Val: {t2-t1} | Test: {n-t2}")

    print("\n[3/3] Training Attention-LSTM...")
    fc = WeatherForecaster(input_dim=6, hidden_dim_1=64, hidden_dim_2=32, dropout=0.25)
    start = time.time()
    result = fc.train(X[:t1], y[:t1], X[t1:t2], y[t1:t2], epochs=50, batch_size=32, patience=10)
    print(f"\n  Completed in {time.time()-start:.1f}s | Best val loss: {result['best_val_loss']:.6f}")

    metrics = fc.evaluate(X[t2:], y[t2:], pp, pp.FEATURES)
    print("\n  Test Metrics (original scale):")
    for feat, m in metrics.items():
        print(f"    {feat}: MAE={m['MAE']:.3f}, RMSE={m['RMSE']:.3f}")

    save_dir = Path(__file__).parent.parent / "saved_models" / "lstm"
    save_dir.mkdir(parents=True, exist_ok=True)
    fc.save(str(save_dir / "attention_lstm.pt"))
    pp.save_scalers(str(save_dir / "scalers.pkl"))
    return metrics


def train_crop_model():
    print("\n" + "=" * 60)
    print("TRAINING: Crop Recommender (XGBoost + DNN Ensemble)")
    print("=" * 60)
    from data.weather_pipeline import CropDataGenerator
    from models_pkg.crop.crop_recommender import CropRecommender

    print("\n[1/2] Generating crop dataset from agricultural KB...")
    gen = CropDataGenerator()
    df = gen.generate(n_samples=8500)
    print(f"  {len(df)} samples, {df['crop'].nunique()} crops")

    print("\n[2/2] Training ensemble...")
    rec = CropRecommender(xgb_weight=0.6, dnn_weight=0.4)
    start = time.time()
    result = rec.train(df, epochs=30, batch_size=64)
    print(f"\n  Completed in {time.time()-start:.1f}s")

    save_path = str(Path(__file__).parent.parent / "saved_models" / "crop")
    rec.save(save_path)

    print("\n  Test predictions:")
    for loc, params in [("Mandya (kharif)", (26, 70, 120, 8, "red_laterite", "kharif")),
                         ("Delhi (rabi)", (15, 55, 40, 6, "alluvial", "rabi")),
                         ("Mangalore (kharif)", (28, 82, 250, 12, "loamy", "kharif"))]:
        preds = rec.predict(*params)
        crops_str = ', '.join(f'{p["crop"]} ({p["confidence"]*100:.0f}%)' for p in preds)
        print(f"    {loc}: {crops_str}")
    return result


def test_full_pipeline():
    print("\n" + "=" * 60)
    print("TESTING: Full Agent Pipeline with LIVE Weather")
    print("=" * 60)
    import asyncio
    from agent.reasoning_engine import AgentReasoningEngine

    agent = AgentReasoningEngine()

    async def run():
        queries = [
            "What crop should I plant in Mandya next month?",
            "Pesticide advice for rice in Shimoga",
            "Fertilizer schedule for wheat at flowering stage",
            "Irrigation plan for maize in Pune",
            "Weather forecast for Bangalore next week",
        ]
        for q in queries:
            print(f"\n{'='*50}")
            print(f"Q: {q}")
            r = await agent.process_query(q)
            if r["type"] == "clarification":
                print(f"  -> Clarification: {r['response']}")
            else:
                print(f"  Intent: {r['intent']} | Location: {r.get('location','')}")
                w = r.get("weather", {})
                print(f"  Weather source: {w.get('source','?')} | Days: {w.get('forecast_days','?')}")
                s = w.get("summary", {})
                print(f"  Conditions: {s.get('temp_min','?')}-{s.get('temp_max','?')}°C, {s.get('humidity_avg','?')}% hum, {s.get('rainfall_total','?')}mm rain")
                for mod, data in r.get("advisories", {}).items():
                    if mod == "crop":
                        for rc in data.get("recommendations", [])[:3]:
                            print(f"    Crop: {rc['crop']} ({rc.get('confidence',0)*100:.0f}%)")
                    elif mod == "pesticide":
                        for p in data.get("risks", {}).get("pests", [])[:2]:
                            print(f"    Pest risk: {p['pest']} ({p['probability']*100:.0f}%)")
                        for sw in data.get("spray_windows", [])[:3]:
                            print(f"    Spray {sw['date']}: {'SAFE' if sw['safe'] else 'AVOID'}")
                    elif mod == "irrigation":
                        ia = data.get("analysis", {})
                        print(f"    ET0={data.get('et0','?')}, Deficit={ia.get('irrigation_deficit_mm','?')}mm/day")
                    elif mod == "weather":
                        for d in data.get("daily", [])[:3]:
                            print(f"    {d['date']}: {d['temp_min']}-{d['temp_max']}°C, {d['rainfall']}mm rain")
            agent.dialogue.reset()

    asyncio.run(run())


if __name__ == "__main__":
    print("KrishiMitra Training Pipeline")
    print("Using REAL weather data from Open-Meteo (free, no API key)")
    print("=" * 60)
    train_weather_model()
    train_crop_model()
    test_full_pipeline()
    print("\n" + "=" * 60)
    print("ALL DONE! Start the server with: python api/main.py")
    print("=" * 60)
