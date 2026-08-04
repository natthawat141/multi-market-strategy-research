"""Gradient-boosted tree model (SPEC.md section 8.2, primary tabular model).

LightGBM is used when importable; XGBoost is the documented fallback
(SPEC.md / CLAUDE.md: "XGBoost is an acceptable documented fallback").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from frtbot.models.base import CountryModel
from frtbot.models.dataset import drop_incomplete, feature_columns


class TreeModel(CountryModel):
    name = "tree"

    def __init__(self, seed: int = 42, num_leaves: int = 15, n_estimators: int = 300):
        self.seed = seed
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.feature_cols: list[str] | None = None
        self.model = None
        self.backend: str | None = None

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> TreeModel:
        self.feature_cols = feature_columns(train_df)
        clean_train = drop_incomplete(train_df, self.feature_cols)
        X_train = clean_train[self.feature_cols].to_numpy()
        y_train = clean_train["label_regression"].to_numpy()

        eval_set = None
        if val_df is not None:
            clean_val = drop_incomplete(val_df, self.feature_cols)
            if len(clean_val) > 0:
                eval_set = (clean_val[self.feature_cols].to_numpy(), clean_val["label_regression"].to_numpy())

        try:
            import lightgbm as lgb

            self.backend = "lightgbm"
            self.model = lgb.LGBMRegressor(
                num_leaves=self.num_leaves,
                n_estimators=self.n_estimators,
                learning_rate=0.03,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.seed,
                verbosity=-1,
            )
            fit_kwargs = {}
            if eval_set is not None:
                fit_kwargs["eval_X"] = eval_set[0]
                fit_kwargs["eval_y"] = eval_set[1]
                fit_kwargs["callbacks"] = [lgb.early_stopping(30, verbose=False)]
            self.model.fit(X_train, y_train, **fit_kwargs)
        except ImportError:
            import xgboost as xgb

            self.backend = "xgboost"
            self.model = xgb.XGBRegressor(
                max_depth=4,
                n_estimators=self.n_estimators,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.seed,
            )
            if eval_set is not None:
                self.model.fit(X_train, y_train, eval_set=[eval_set], verbose=False)
            else:
                self.model.fit(X_train, y_train)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None or self.feature_cols is None:
            raise RuntimeError(f"{self.name} must be fit before predict")
        valid = df[self.feature_cols].notna().all(axis=1)
        scores = pd.Series(np.nan, index=df.index, name=self.name)
        if valid.any():
            scores.loc[valid] = self.model.predict(df.loc[valid, self.feature_cols].to_numpy())
        return scores
