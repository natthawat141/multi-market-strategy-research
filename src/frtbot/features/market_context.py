"""Market-context features (SPEC.md section 6.3).

"Local market trend and momentum" is not duplicated here: at the country-level
vertical slice, a market's own trend/momentum *is* `price_trend.py`'s output
for that market's proxy (there is no separate stock universe yet - see M3).
This module covers what is inherently cross-market: relative momentum rank,
rolling correlation to a global benchmark, and a simple risk-regime signal.
Local-currency-vs-THB return is `frtbot.data.fx.local_fx_return`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from frtbot.features.normalize import cross_sectional_percentile_rank


def cross_market_relative_momentum_rank(momentum_by_market: dict[str, pd.Series]) -> pd.DataFrame:
    """Same-date percentile rank of each market's momentum among available markets."""
    wide = pd.DataFrame(momentum_by_market)
    return cross_sectional_percentile_rank(wide).add_suffix("_rel_mom_rank")


def global_equal_weight_return(return_by_market: dict[str, pd.Series]) -> pd.Series:
    """Equal-weight average daily return across whichever markets have data on a date."""
    wide = pd.DataFrame(return_by_market)
    return wide.mean(axis=1, skipna=True).rename("global_ew_return")


def global_risk_regime(
    global_return: pd.Series, vol_window: int = 21, regime_window: int = 252
) -> pd.Series:
    """Rolling global volatility, z-scored against its own trailing history.

    Positive values indicate an elevated-volatility ("risk-off") regime
    relative to the trailing `regime_window`-day history; both windows use
    only data up to and including the current date.
    """
    rolling_vol = global_return.rolling(vol_window).std(ddof=0) * np.sqrt(252)
    mean = rolling_vol.rolling(regime_window).mean()
    std = rolling_vol.rolling(regime_window).std(ddof=0).replace(0.0, np.nan)
    return ((rolling_vol - mean) / std).rename("global_risk_regime_z")


def rolling_corr_with_global(
    return_by_market: dict[str, pd.Series], global_return: pd.Series, window: int = 63
) -> pd.DataFrame:
    """Rolling correlation of each market's return with the global equal-weight return.

    `global_return` is reindexed onto each market's own trading-date index
    before correlating. Different exchanges close on different local
    holidays, so correlating against the shared union index would inject
    single-day gaps into an otherwise-complete market series; a single NaN
    anywhere in a `Series.rolling().corr()` window poisons that entire
    window's result, which made this feature almost always NaN before the
    reindex. `global_return` is itself NaN-free on any date a market traded
    (SPEC.md 6.3 "cross-market" convention: at least one of the five markets
    is open on all but a handful of universal holidays), so this reindex
    does not introduce new gaps.
    """
    out = {}
    for key, series in return_by_market.items():
        aligned_global = global_return.reindex(series.index)
        out[f"{key}_corr_global_{window}d"] = series.rolling(window).corr(aligned_global)
    return pd.DataFrame(out)
