"""Shared model interface: every model turns a features panel into a score.

Higher score = more attractive for the next `horizon` bars. All signal models
(baselines, tree, MLP, transparent trend) implement this so the identical
portfolio-construction and backtest pipeline can consume any of them, per
SPEC.md acceptance criteria #5 ("Tree, MLP, transparent trend, and simple
benchmarks are compared on identical dates").
"""

from __future__ import annotations

import abc

import pandas as pd


class CountryModel(abc.ABC):
    name: str

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> CountryModel:
        """Default: stateless (rule-based) models need no fitting."""
        return self

    @abc.abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return a score Series aligned to `df.index` (MultiIndex date, market)."""


def same_date_percentile_rank(scores: pd.Series) -> pd.Series:
    """Percentile-rank a (date, market)-indexed score within each date (0-1, NaN-safe)."""
    return scores.groupby(level="date").rank(pct=True)
