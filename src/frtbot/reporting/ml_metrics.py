"""Primary ML metrics (SPEC.md section 11 "Primary ML metrics").

Scoped to what is meaningful at the five-market country-level vertical slice:
rank information coefficient and top-quintile hit rate. Calibration/Brier
score and prediction-stability-by-regime are more informative with the larger
per-date cross-sections available once stock-mode (M3) adapters land, and are
deferred - see README limitations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rank_information_coefficient(score: pd.Series, forward_return: pd.Series) -> float:
    """Mean per-date Spearman correlation between a (date, market)-indexed score and label."""
    df = pd.DataFrame({"score": score, "y": forward_return}).dropna()
    if df.empty:
        return float("nan")

    def _ic(group: pd.DataFrame) -> float:
        if len(group) < 2:
            return float("nan")
        return group["score"].corr(group["y"], method="spearman")

    per_date = df.groupby(level="date").apply(_ic)
    return float(per_date.mean())


def top_quintile_hit_rate(score: pd.Series, forward_return: pd.Series, quintile: float = 0.2) -> float:
    """Fraction of dates where the top-`quintile`-ranked-by-score market(s) were also
    actually in the top `quintile` of realized forward return that date."""
    df = pd.DataFrame({"score": score, "y": forward_return}).dropna()
    if df.empty:
        return float("nan")

    def _hit(group: pd.DataFrame) -> float:
        n = len(group)
        if n < 2:
            return np.nan
        k = max(1, int(round(n * quintile)))
        predicted_top = set(group.sort_values("score", ascending=False).index[:k])
        actual_top = set(group.sort_values("y", ascending=False).index[:k])
        return len(predicted_top & actual_top) / k

    per_date = df.groupby(level="date").apply(_hit)
    return float(per_date.mean())
