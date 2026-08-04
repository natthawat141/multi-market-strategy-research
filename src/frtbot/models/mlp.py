"""Small MLP regressor (SPEC.md section 8.2): regularization, early stopping, fixed seed."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from frtbot.models.base import CountryModel
from frtbot.models.dataset import drop_incomplete, feature_columns


class _MLPModule:
    """Lazily built so importing this module does not require torch at import time."""

    def __new__(cls, input_dim: int, hidden: tuple[int, ...], dropout: float):
        import torch.nn as nn

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)


class MLPModel(CountryModel):
    name = "mlp"

    def __init__(
        self,
        seed: int = 42,
        hidden: tuple[int, ...] = (32, 16),
        dropout: float = 0.1,
        weight_decay: float = 1e-4,
        lr: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 15,
    ):
        self.seed = seed
        self.hidden = hidden
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.feature_cols: list[str] | None = None
        self.scaler: StandardScaler | None = None
        self.model = None

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> MLPModel:
        import torch

        torch.manual_seed(self.seed)

        self.feature_cols = feature_columns(train_df)
        clean_train = drop_incomplete(train_df, self.feature_cols)
        if val_df is not None:
            clean_val = drop_incomplete(val_df, self.feature_cols)
        else:
            clean_val = clean_train.sample(frac=0.2, random_state=self.seed)
            clean_train = clean_train.drop(clean_val.index)

        self.scaler = StandardScaler().fit(clean_train[self.feature_cols].to_numpy())
        X_train = self.scaler.transform(clean_train[self.feature_cols].to_numpy())
        y_train = clean_train["label_regression"].to_numpy(dtype=np.float32)
        X_val = self.scaler.transform(clean_val[self.feature_cols].to_numpy())
        y_val = clean_val["label_regression"].to_numpy(dtype=np.float32)

        model = _MLPModule(len(self.feature_cols), self.hidden, self.dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()

        X_train_t = torch.as_tensor(X_train, dtype=torch.float32)
        y_train_t = torch.as_tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_val_t = torch.as_tensor(X_val, dtype=torch.float32)
        y_val_t = torch.as_tensor(y_val, dtype=torch.float32).unsqueeze(1)

        rng = np.random.default_rng(self.seed)
        n = X_train_t.shape[0]
        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        for _epoch in range(self.max_epochs):
            model.train()
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                batch_X = X_train_t[idx]
                batch_y = y_train_t[idx]
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(X_val_t), y_val_t).item()

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model = model
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        import torch

        if self.model is None or self.scaler is None or self.feature_cols is None:
            raise RuntimeError(f"{self.name} must be fit before predict")
        valid = df[self.feature_cols].notna().all(axis=1)
        scores = pd.Series(np.nan, index=df.index, name=self.name)
        if valid.any():
            X = self.scaler.transform(df.loc[valid, self.feature_cols].to_numpy())
            with torch.no_grad():
                preds = self.model(torch.as_tensor(X, dtype=torch.float32)).squeeze(1).numpy()
            scores.loc[valid] = preds
        return scores
