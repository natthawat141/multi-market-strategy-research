"""Country allocation (SPEC.md section 9.1).

- Consider only markets with positive predicted THB excess return and valid data.
- Select up to the top three eligible markets (by ensemble score).
- Start from inverse 63-day volatility weights.
- Cap a country at 40%.
- Unallocated capital remains in the configured cash proxy.
"""

from __future__ import annotations

import pandas as pd

from frtbot.config import PortfolioConfig


def apply_country_cap(weights: pd.Series, cap: float) -> pd.Series:
    """Proportionally allocate `weights` (assumed to already sum to <= 1) under a per-name cap.

    Any budget that cannot be redistributed because every remaining name is
    already at the cap is left unallocated (the caller treats the shortfall
    as residual cash) - a standard iterative waterfilling cap.
    """
    w = weights.astype(float)
    fixed = pd.Series(0.0, index=w.index)
    free_names = list(w.index)
    remaining_budget = float(w.sum())

    while free_names:
        free_raw_sum = w[free_names].sum()
        if free_raw_sum <= 0:
            break
        proportional = {n: remaining_budget * (w[n] / free_raw_sum) for n in free_names}
        over = [n for n in free_names if proportional[n] > cap + 1e-12]
        if not over:
            for n in free_names:
                fixed[n] = proportional[n]
            break
        for n in over:
            fixed[n] = cap
            remaining_budget -= cap
            free_names.remove(n)

    return fixed


def construct_country_weights(
    ensemble_score: pd.Series,
    predicted_excess_return: pd.Series,
    vol_63d: pd.Series,
    config: PortfolioConfig,
    valid_markets: set[str] | None = None,
) -> pd.Series:
    """Return a weight Series over selected markets plus `config.cash_key`, summing to 1.0."""
    candidates = [
        m
        for m in ensemble_score.index
        if (valid_markets is None or m in valid_markets)
        and pd.notna(ensemble_score.get(m))
        and pd.notna(predicted_excess_return.get(m))
        and predicted_excess_return[m] > 0
    ]
    eligible_sorted = sorted(candidates, key=lambda m: (-ensemble_score[m], m))
    selected = eligible_sorted[: config.top_n_countries]

    if not selected:
        return pd.Series({config.cash_key: 1.0})

    inv_vol = {}
    for m in selected:
        v = vol_63d.get(m)
        if pd.notna(v) and v > 0:
            inv_vol[m] = 1.0 / v
    missing = [m for m in selected if m not in inv_vol]
    if missing:
        fallback = (sum(inv_vol.values()) / len(inv_vol)) if inv_vol else 1.0
        for m in missing:
            inv_vol[m] = fallback

    raw = pd.Series(inv_vol).reindex(selected)
    normalized = raw / raw.sum()
    capped = apply_country_cap(normalized, config.max_country_weight)

    residual = 1.0 - float(capped.sum())
    out = capped.copy()
    if residual > 1e-9:
        out[config.cash_key] = residual
    return out
