"""Required simple baselines (SPEC.md section 8.1).

`EqualWeightModel` and `RuleMomentumModel` are stateless rules. `BuyAndHold`
is not a per-date scoring model at all (it never re-ranks anything) and is
computed directly as a benchmark equity curve in `frtbot.reporting.metrics`,
not here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from frtbot.models.base import CountryModel
from frtbot.models.dataset import drop_incomplete, feature_columns


class EqualWeightModel(CountryModel):
    """No ranking signal: every valid market gets an identical positive score."""

    name = "equal_weight"

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=df.index, name=self.name)


class RuleMomentumModel(CountryModel):
    """Absolute-plus-relative momentum rule (SPEC.md 8.1), distinct from the
    section 8.2 transparent trend score used in the ensemble."""

    name = "rule_momentum"

    def predict(self, df: pd.DataFrame) -> pd.Series:
        abs_mom_positive = (df["thb_ret_63d"] > 0).astype(float)
        rel_rank = df["rel_mom_rank"].fillna(0.5)
        score = 0.5 * rel_rank + 0.5 * abs_mom_positive
        return score.rename(self.name)


class RidgeBaselineModel(CountryModel):
    """Regularized linear baseline (SPEC.md 8.1). Scaler/model fit on train only."""

    name = "linear_ridge"

    def __init__(self, alpha: float = 1.0, seed: int = 42):
        self.alpha = alpha
        self.seed = seed
        self.feature_cols: list[str] | None = None
        self.scaler: StandardScaler | None = None
        self.model: Ridge | None = None

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> RidgeBaselineModel:
        self.feature_cols = feature_columns(train_df)
        clean = drop_incomplete(train_df, self.feature_cols)
        self.scaler = StandardScaler().fit(clean[self.feature_cols].to_numpy())
        X = self.scaler.transform(clean[self.feature_cols].to_numpy())
        y = clean["label_regression"].to_numpy()
        self.model = Ridge(alpha=self.alpha, random_state=self.seed).fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None or self.scaler is None or self.feature_cols is None:
            raise RuntimeError(f"{self.name} must be fit before predict")
        valid = df[self.feature_cols].notna().all(axis=1)
        scores = pd.Series(np.nan, index=df.index, name=self.name)
        if valid.any():
            X = self.scaler.transform(df.loc[valid, self.feature_cols].to_numpy())
            scores.loc[valid] = self.model.predict(X)
        return scores
