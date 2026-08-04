"""Transparent rule-based trend score (SPEC.md section 8.2).

A same-date cross-sectional average of percentile ranks across a small,
interpretable set of trend features - no fitted parameters, fully
inspectable. Distinct from `RuleMomentumModel` (SPEC.md 8.1 baseline), which
is a simpler absolute+relative momentum filter.
"""

from __future__ import annotations

import pandas as pd

from frtbot.models.base import CountryModel

TREND_COLUMNS = ("dist_sma_50", "dist_sma_200", "mom_12_1", "donchian_20_position", "rsi_14")


class TransparentTrendModel(CountryModel):
    name = "transparent_trend"

    def __init__(self, columns: tuple[str, ...] = TREND_COLUMNS):
        self.columns = columns

    def predict(self, df: pd.DataFrame) -> pd.Series:
        ranks = [df[col].groupby(level="date").rank(pct=True) for col in self.columns]
        score = pd.concat(ranks, axis=1).mean(axis=1)
        return score.rename(self.name)
