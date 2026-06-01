"""
Crop Recommendation Engine
Ensemble model: XGBoost + Neural Network with meta-learner.
Inputs: LSTM weather forecast + soil + season features.
Output: Top-K crops with calibrated confidence scores.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os, pickle, json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from configs.settings import crop_cfg


class CropDNN(nn.Module):
    """3-layer Deep Neural Network for crop classification."""
    def __init__(self, input_dim: int, num_classes: int,
                 hidden_layers: List[int] = [128, 64, 32], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CropDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class CropRecommender:
    """
    Ensemble crop recommendation system.
    Uses XGBoost for robust tree-based prediction + DNN for pattern learning.
    Final prediction: weighted average of both models' probabilities.
    """
    def __init__(self, xgb_weight: float = 0.6, dnn_weight: float = 0.4):
        self.xgb_model = None
        self.dnn_model = None
        self.label_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
        self.soil_encoder = LabelEncoder()
        self.season_encoder = LabelEncoder()
        self.xgb_weight = xgb_weight
        self.dnn_weight = dnn_weight
        self.num_classes = 0
        self.feature_names = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _prepare_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Encode categorical features and scale numerical."""
        df = df.copy()
        if fit:
            df["soil_encoded"] = self.soil_encoder.fit_transform(df["soil_type"])
            df["season_encoded"] = self.season_encoder.fit_transform(df["season"])
        else:
            # Handle unseen categories
            df["soil_encoded"] = df["soil_type"].apply(
                lambda x: self.soil_encoder.transform([x])[0] if x in self.soil_encoder.classes_ else 0
            )
            df["season_encoded"] = df["season"].apply(
                lambda x: self.season_encoder.transform([x])[0] if x in self.season_encoder.classes_ else 0
            )

        feature_cols = ["temp_avg", "humidity_avg", "rainfall_monthly",
                        "wind_speed_avg", "soil_encoded", "season_encoded",
                        "elevation_m", "soil_ph"]
        self.feature_names = feature_cols
        X = df[feature_cols].values.astype(np.float32)
        if fit:
            X = self.feature_scaler.fit_transform(X)
        else:
            X = self.feature_scaler.transform(X)
        return X

    def train(self, df: pd.DataFrame, epochs: int = 30, batch_size: int = 64) -> Dict:
        """Train both XGBoost and DNN models."""
        print("\n[CropRecommender] Training ensemble model...")

        # Encode labels
        y = self.label_encoder.fit_transform(df["crop"])
        self.num_classes = len(self.label_encoder.classes_)
        X = self._prepare_features(df, fit=True)

        # Split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # ---- XGBoost ----
        print("  Training XGBoost...")
        self.xgb_model = xgb.XGBClassifier(
            max_depth=6, learning_rate=0.1, n_estimators=200,
            objective="multi:softprob", num_class=self.num_classes,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", verbosity=0, random_state=42
        )
        self.xgb_model.fit(X_train, y_train,
                           eval_set=[(X_test, y_test)], verbose=False)
        xgb_pred = self.xgb_model.predict(X_test)
        xgb_f1 = f1_score(y_test, xgb_pred, average="macro")
        print(f"  XGBoost Macro F1: {xgb_f1:.4f}")

        # ---- DNN ----
        print("  Training DNN...")
        self.dnn_model = CropDNN(
            input_dim=X.shape[1], num_classes=self.num_classes
        ).to(self.device)
        optimizer = torch.optim.Adam(self.dnn_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        train_ds = CropDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        self.dnn_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.dnn_model(Xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs} Loss: {total_loss/len(train_loader):.4f}")

        # DNN evaluation
        self.dnn_model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(self.device)
            dnn_out = self.dnn_model(X_test_t)
            dnn_pred = dnn_out.argmax(dim=1).cpu().numpy()
        dnn_f1 = f1_score(y_test, dnn_pred, average="macro")
        print(f"  DNN Macro F1: {dnn_f1:.4f}")

        # Ensemble evaluation
        xgb_probs = self.xgb_model.predict_proba(X_test)
        dnn_probs = torch.softmax(dnn_out, dim=1).cpu().numpy()
        ensemble_probs = self.xgb_weight * xgb_probs + self.dnn_weight * dnn_probs
        ensemble_pred = ensemble_probs.argmax(axis=1)
        ensemble_f1 = f1_score(y_test, ensemble_pred, average="macro")
        print(f"  Ensemble Macro F1: {ensemble_f1:.4f}")

        return {
            "xgb_f1": round(xgb_f1, 4),
            "dnn_f1": round(dnn_f1, 4),
            "ensemble_f1": round(ensemble_f1, 4),
            "num_classes": self.num_classes,
            "num_samples": len(df),
        }

    def predict(self, temp_avg: float, humidity_avg: float, rainfall_monthly: float,
                wind_speed_avg: float, soil_type: str, season: str,
                elevation_m: float = 500, soil_ph: float = 6.5,
                top_k: int = 3) -> List[Dict]:
        """Predict top-K crops with confidence scores."""
        input_df = pd.DataFrame([{
            "temp_avg": temp_avg, "humidity_avg": humidity_avg,
            "rainfall_monthly": rainfall_monthly, "wind_speed_avg": wind_speed_avg,
            "soil_type": soil_type, "season": season,
            "elevation_m": elevation_m, "soil_ph": soil_ph,
        }])
        X = self._prepare_features(input_df, fit=False)

        # XGBoost probabilities
        xgb_probs = self.xgb_model.predict_proba(X)[0]

        # DNN probabilities
        self.dnn_model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            dnn_out = self.dnn_model(X_t)
            dnn_probs = torch.softmax(dnn_out, dim=1).cpu().numpy()[0]

        # Ensemble
        probs = self.xgb_weight * xgb_probs + self.dnn_weight * dnn_probs

        # Top-K
        top_indices = np.argsort(probs)[::-1][:top_k]
        results = []
        for idx in top_indices:
            crop_name = self.label_encoder.inverse_transform([idx])[0]
            results.append({
                "crop": crop_name,
                "confidence": round(float(probs[idx]), 4),
                "xgb_score": round(float(xgb_probs[idx]), 4),
                "dnn_score": round(float(dnn_probs[idx]), 4),
            })
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.xgb_model.save_model(os.path.join(path, "xgb_model.json"))
        torch.save(self.dnn_model.state_dict(), os.path.join(path, "dnn_model.pt"))
        with open(os.path.join(path, "encoders.pkl"), "wb") as f:
            pickle.dump({
                "label_encoder": self.label_encoder,
                "feature_scaler": self.feature_scaler,
                "soil_encoder": self.soil_encoder,
                "season_encoder": self.season_encoder,
                "num_classes": self.num_classes,
                "feature_names": self.feature_names,
            }, f)
        print(f"  Crop model saved to {path}")

    def load(self, path: str):
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(os.path.join(path, "xgb_model.json"))
        with open(os.path.join(path, "encoders.pkl"), "rb") as f:
            enc = pickle.load(f)
            self.label_encoder = enc["label_encoder"]
            self.feature_scaler = enc["feature_scaler"]
            self.soil_encoder = enc["soil_encoder"]
            self.season_encoder = enc["season_encoder"]
            self.num_classes = enc["num_classes"]
            self.feature_names = enc["feature_names"]
        self.dnn_model = CropDNN(
            input_dim=len(self.feature_names), num_classes=self.num_classes
        ).to(self.device)
        self.dnn_model.load_state_dict(
            torch.load(os.path.join(path, "dnn_model.pt"), map_location=self.device, weights_only=True)
        )
        print(f"  Crop model loaded from {path}")
