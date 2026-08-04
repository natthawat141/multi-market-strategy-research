"""Country-level labels (SPEC.md section 7.1).

Regression label: `next_21d_country_return_thb - next_21d_cash_return_thb`.
Classification label: 1 when that excess return is positive, else 0.

These are forward-looking by construction (they use price data *after* date
`t`) and must never be used as model inputs or preprocessing statistics -
only as the training/evaluation target, computed once and then handled by the
purge/embargo logic in `frtbot.backtest.splits` before any walk-forward split
boundary.
"""

from __future__ import annotations

import pandas as pd


def next_horizon_country_return_thb(thb_close: pd.Series, horizon: int = 21) -> pd.Series:
    """Forward THB total return from date `t` to `t + horizon` trading bars.

    `thb_close.shift(-horizon)` pulls the future price back onto date `t`;
    the last `horizon` rows are NaN because no future price exists yet.
    """
    forward_price = thb_close.shift(-horizon)
    return (forward_price / thb_close - 1.0).rename(f"next_{horizon}d_return_thb")


def next_horizon_cash_return_thb(
    index: pd.DatetimeIndex, cash_annual_rate: float, horizon: int = 21
) -> pd.Series:
    """Constant compounded cash-proxy return over `horizon` trading bars (SPEC.md 9.1)."""
    trading_days_per_year = 252
    rate = (1.0 + cash_annual_rate) ** (horizon / trading_days_per_year) - 1.0
    return pd.Series(rate, index=index, name=f"next_{horizon}d_cash_return_thb")


def country_regression_label(
    thb_close: pd.Series, cash_annual_rate: float, horizon: int = 21
) -> pd.Series:
    """`next_21d_country_return_thb - next_21d_cash_return_thb` (SPEC.md 7.1 primary label)."""
    country_ret = next_horizon_country_return_thb(thb_close, horizon)
    cash_ret = next_horizon_cash_return_thb(thb_close.index, cash_annual_rate, horizon)
    return (country_ret - cash_ret).rename(f"label_excess_return_{horizon}d")


def country_classification_label(regression_label: pd.Series) -> pd.Series:
    """1 when the excess-return label is positive, else 0; NaN preserved where undefined."""
    out = (regression_label > 0).astype("Int64")
    out = out.where(regression_label.notna())
    return out.rename("label_excess_positive")
