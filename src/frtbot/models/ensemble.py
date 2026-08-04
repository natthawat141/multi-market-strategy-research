"""Frozen ensemble formula (SPEC.md section 8.4).

    ensemble_score = 0.40 * tree_rank + 0.30 * mlp_rank + 0.30 * trend_rank

Fixed research starting weights, never optimized on the test set. A model
that fails validation is excluded and the remaining weights renormalized,
with the exclusion reported (not hidden).
"""

from __future__ import annotations

import pandas as pd

from frtbot.config import EnsembleConfig
from frtbot.models.base import same_date_percentile_rank


class EnsembleResult:
    def __init__(self, score: pd.Series, weights_used: dict[str, float], excluded: list[str]):
        self.score = score
        self.weights_used = weights_used
        self.excluded = excluded


def combine_ensemble(
    scores: dict[str, pd.Series],
    ensemble_config: EnsembleConfig,
    excluded: set[str] | None = None,
) -> EnsembleResult:
    excluded = set(excluded or set())
    frozen_weights = {
        "tree": ensemble_config.tree_weight,
        "mlp": ensemble_config.mlp_weight,
        "transparent_trend": ensemble_config.trend_weight,
    }
    missing = [k for k in frozen_weights if k not in scores]
    excluded = excluded | set(missing)

    active = {k: w for k, w in frozen_weights.items() if k not in excluded}
    total = sum(active.values())
    if total <= 0:
        raise ValueError(f"No active ensemble models remain (excluded={sorted(excluded)})")
    renormalized = {k: w / total for k, w in active.items()}

    ranked = {k: same_date_percentile_rank(scores[k]) for k in active}
    combined = sum(renormalized[k] * ranked[k] for k in active)
    combined = combined.rename("ensemble_score")

    return EnsembleResult(combined, renormalized, sorted(excluded))
