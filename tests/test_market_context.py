from __future__ import annotations

import numpy as np
import pandas as pd

from frtbot.features.market_context import global_equal_weight_return, rolling_corr_with_global


def test_rolling_corr_with_global_survives_misaligned_calendars():
    """Regression test: real exchanges close on different local holidays, so a naive
    rolling correlation against a union-indexed global return goes almost entirely NaN
    (a single missing day poisons an entire pandas rolling-corr window). Two markets on
    deliberately offset calendars (like real exchanges) must still produce a mostly
    non-NaN rolling correlation once warmed up."""
    rng = np.random.default_rng(0)

    dates_a = pd.bdate_range("2018-01-01", periods=400)
    dates_b = dates_a.delete([5, 40, 90, 150, 200, 250, 300, 350])  # market B has extra local holidays

    ret_a = pd.Series(rng.normal(0, 0.01, len(dates_a)), index=dates_a)
    ret_b = pd.Series(rng.normal(0, 0.01, len(dates_b)), index=dates_b)

    global_ret = global_equal_weight_return({"A": ret_a, "B": ret_b})
    corr = rolling_corr_with_global({"A": ret_a, "B": ret_b}, global_ret, window=63)

    warmed_up = corr.iloc[70:]
    assert warmed_up["A_corr_global_63d"].isna().mean() < 0.05
    assert warmed_up["B_corr_global_63d"].isna().mean() < 0.05
