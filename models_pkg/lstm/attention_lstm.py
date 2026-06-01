"""
Attention-LSTM Weather Forecaster
Stacked LSTM with temporal self-attention for multivariate weather prediction.
Supports multi-horizon forecasting (1, 7, 14, 30 days).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import os, json


class TemporalAttention(nn.Module):
    """Self-attention mechanism over LSTM hidden states.
    Learns which past time-steps are most important for each forecast.
    α_t = softmax(W_a · tanh(W_h · h_t + b))
    """
    def __init__(self, hidden_dim: int, attention_dim: int = 32):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim, attention_dim, bias=True)
        self.W_a = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_dim)
        Returns:
            context: (batch, hidden_dim) - attention-weighted representation
            weights: (batch, seq_len) - attention weights (interpretable)
        """
        energy = torch.tanh(self.W_h(lstm_output))       # (batch, seq, attn_dim)
        scores = self.W_a(energy).squeeze(-1)             # (batch, seq)
        weights = F.softmax(scores, dim=-1)               # (batch, seq)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)  # (batch, hidden)
        return context, weights


class AttentionLSTM(nn.Module):
    """
    Stacked LSTM (64→32) with temporal self-attention for weather forecasting.

    Architecture:
        Input (batch, lookback, features)
        → LSTM Layer 1 (64 units, return sequences)
        → Dropout(0.25)
        → LSTM Layer 2 (32 units, return sequences)
        → Temporal Self-Attention
        → Dense → Output (features)
    """
    def __init__(self, input_dim: int = 6, hidden_dim_1: int = 64,
                 hidden_dim_2: int = 32, dropout: float = 0.25,
                 output_dim: int = 6):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim_1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim_1, hidden_dim_2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.attention = TemporalAttention(hidden_dim_2)
        self.fc = nn.Linear(hidden_dim_2, output_dim)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, lookback, features)
        Returns:
            prediction: (batch, output_dim)
            attention_weights: (batch, lookback)
        """
        out1, _ = self.lstm1(x)
        out1 = self.dropout1(out1)
        out2, _ = self.lstm2(out1)
        out2 = self.dropout2(out2)
        context, attn_weights = self.attention(out2)
        prediction = self.fc(context)
        return prediction, attn_weights


class MultiHorizonAttentionLSTM(nn.Module):
    """
    Multi-horizon variant with separate prediction heads per horizon.
    Shared LSTM backbone, separate dense heads for 1/7/14/30 day forecasts.
    """
    def __init__(self, input_dim: int = 6, hidden_dim_1: int = 64,
                 hidden_dim_2: int = 32, dropout: float = 0.25,
                 output_dim: int = 6, horizons: List[int] = [1, 7, 14, 30]):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim_1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim_1, hidden_dim_2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.attention = TemporalAttention(hidden_dim_2)
        self.horizons = horizons
        self.heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(hidden_dim_2, hidden_dim_2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim_2, output_dim)
            ) for h in horizons
        })
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor, horizon: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        out1, _ = self.lstm1(x)
        out1 = self.dropout1(out1)
        out2, _ = self.lstm2(out1)
        out2 = self.dropout2(out2)
        context, attn_weights = self.attention(out2)
        prediction = self.heads[str(horizon)](context)
        return prediction, attn_weights

    def predict_all_horizons(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Get predictions for all horizons in one forward pass (shared backbone)."""
        out1, _ = self.lstm1(x)
        out1 = self.dropout1(out1)
        out2, _ = self.lstm2(out1)
        out2 = self.dropout2(out2)
        context, attn_weights = self.attention(out2)
        results = {}
        for h in self.horizons:
            results[h] = self.heads[str(h)](context)
        return results, attn_weights


class WeatherDataset(Dataset):
    """PyTorch Dataset for weather sequences."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class WeatherForecaster:
    """
    Complete weather forecasting system wrapping model, training, and inference.
    """
    def __init__(self, input_dim: int = 6, hidden_dim_1: int = 64,
                 hidden_dim_2: int = 32, dropout: float = 0.25,
                 learning_rate: float = 1e-3, huber_delta: float = 1.0,
                 device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AttentionLSTM(
            input_dim=input_dim, hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2, dropout=dropout, output_dim=input_dim
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )
        self.criterion = nn.HuberLoss(delta=huber_delta)
        self.training_history = {"train_loss": [], "val_loss": []}

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32, patience: int = 10) -> Dict:
        """Train with early stopping and learning rate scheduling."""
        train_dataset = WeatherDataset(X_train, y_train)
        val_dataset = WeatherDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                pred, _ = self.model(X_batch)
                loss = self.criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_losses.append(loss.item())

            # Validation
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    pred, _ = self.model(X_batch)
                    loss = self.criterion(pred, y_batch)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.6f}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        return {"best_val_loss": best_val_loss, "epochs_trained": epoch + 1}

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict and return (predictions, attention_weights)."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            pred, attn = self.model(X_tensor)
            return pred.cpu().numpy(), attn.cpu().numpy()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 preprocessor=None, feature_names: List[str] = None) -> Dict:
        """Compute MAE, MSE, RMSE per feature on original scale."""
        pred, attn = self.predict(X_test)
        if preprocessor:
            pred_orig = preprocessor.inverse_transform(pred)
            true_orig = preprocessor.inverse_transform(y_test)
        else:
            pred_orig = pred
            true_orig = y_test

        features = feature_names or [f"feature_{i}" for i in range(pred.shape[1])]
        metrics = {}
        for i, feat in enumerate(features):
            errors = pred_orig[:, i] - true_orig[:, i]
            metrics[feat] = {
                "MAE": round(float(np.mean(np.abs(errors))), 4),
                "MSE": round(float(np.mean(errors ** 2)), 4),
                "RMSE": round(float(np.sqrt(np.mean(errors ** 2))), 4),
            }
        return metrics

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "history": self.training_history,
        }, path)
        print(f"  Model saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.training_history = checkpoint.get("history", {})
        print(f"  Model loaded from {path}")
