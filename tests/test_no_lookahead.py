from __future__ import annotations

import numpy as np
import pandas as pd

from frtbot.backtest.engine import cash_daily_return_series, run_backtest
from frtbot.backtest.splits import generate_walk_forward_folds
from frtbot.config import (
    CostsConfig,
    MarketEntry,
    MarketsConfig,
    WalkForwardConfig,
)
from frtbot.labels.country import country_regression_label, next_horizon_country_return_thb


def test_forward_label_matches_manual_future_return():
    dates = pd.bdate_range("2018-01-01", periods=60)
    close = pd.Series(np.linspace(100, 159, 60), index=dates)  # +1 per day
    fwd = next_horizon_country_return_thb(close, horizon=21)

    t0 = dates[10]
    expected = close.loc[dates[31]] / close.loc[t0] - 1.0
    assert fwd.loc[t0] == expected

    # Last 21 rows have no future price yet and must be NaN, not silently 0 or extrapolated.
    assert fwd.tail(21).isna().all()


def test_regression_label_is_excess_over_cash():
    dates = pd.bdate_range("2018-01-01", periods=60)
    close = pd.Series(100.0, index=dates)  # flat: raw country return is exactly 0
    label = country_regression_label(close, cash_annual_rate=0.05, horizon=21)
    # Flat market: excess return should be the *negative* of the cash return, not zero.
    assert (label.dropna() < 0).all()


def test_walk_forward_folds_are_strictly_chronological_with_embargo():
    dates = pd.bdate_range("2016-01-01", periods=1150)
    cfg = WalkForwardConfig(train_years=2, val_years=1, test_years=1, step_years=1, min_train_years=1, embargo_days=21)
    folds = generate_walk_forward_folds(dates, cfg)
    assert len(folds) > 0
    for fold in folds:
        assert fold.train_dates.max() < fold.val_dates.min()
        assert fold.val_dates.max() < fold.test_dates.min()
        # embargo: at least `embargo_days` trading days are dropped before each boundary,
        # i.e. the gap in *available* dates around the boundary is not simply back-to-back.
        train_end_pos = dates.get_loc(fold.train_dates.max())
        val_start_pos = dates.get_loc(fold.val_dates.min())
        assert val_start_pos - train_end_pos > cfg.embargo_days


def test_walk_forward_folds_are_deterministic():
    dates = pd.bdate_range("2016-01-01", periods=1150)
    cfg = WalkForwardConfig(train_years=2, val_years=1, test_years=1, step_years=1, min_train_years=1, embargo_days=21)
    folds_a = generate_walk_forward_folds(dates, cfg)
    folds_b = generate_walk_forward_folds(dates, cfg)
    assert len(folds_a) == len(folds_b)
    for a, b in zip(folds_a, folds_b, strict=True):
        assert a.train_dates.equals(b.train_dates)
        assert a.val_dates.equals(b.val_dates)
        assert a.test_dates.equals(b.test_dates)


def test_walk_forward_reduces_transparently_when_history_insufficient():
    dates = pd.bdate_range("2016-01-01", periods=100)  # far too short for any real fold
    cfg = WalkForwardConfig()  # default 8y/2y/1y
    folds = generate_walk_forward_folds(dates, cfg)
    assert folds == []


def _toy_markets_config() -> MarketsConfig:
    return MarketsConfig(
        base_currency="THB",
        markets=[
            MarketEntry(
                key="A", name="A", mode="proxy", currency="THB", timezone="Asia/Bangkok",
                exchange_calendar="XBKK", provider="fixture", provider_symbol="A",
                price_type="adjusted_close", adjustment_type="split_dividend_adjusted",
                instrument_class="country_proxy", fx_pair=None,
            ),
            MarketEntry(
                key="B", name="B", mode="proxy", currency="THB", timezone="Asia/Bangkok",
                exchange_calendar="XBKK", provider="fixture", provider_symbol="B",
                price_type="adjusted_close", adjustment_type="split_dividend_adjusted",
                instrument_class="country_proxy", fx_pair=None,
            ),
        ],
        fx=[],
    )


def _toy_costs_config() -> CostsConfig:
    return CostsConfig.model_validate(
        {
            "country_proxy": {"zero": 0, "optimistic": 5, "base": 10, "severe": 25},
            "developed_stock": {"zero": 0, "optimistic": 10, "base": 25, "severe": 50},
            "emerging_stock": {"zero": 0, "optimistic": 20, "base": 50, "severe": 100},
        }
    )


def test_signal_never_trades_on_its_own_close_date():
    """A weight decided at signal date t must not affect the return realized on date t."""
    dates = pd.bdate_range("2018-01-01", periods=10)
    # Both markets flat except a known jump on the signal date itself.
    ret_a = pd.Series(0.0, index=dates)
    ret_b = pd.Series(0.0, index=dates)
    signal_date = dates[3]
    ret_a.loc[signal_date] = 0.50  # huge jump exactly on the signal's own close date

    weights_by_signal_date = {signal_date: pd.Series({"A": 1.0})}
    cash_ret = cash_daily_return_series(dates, 0.0)

    result = run_backtest(
        dates,
        weights_by_signal_date,
        {"A": ret_a, "B": ret_b},
        cash_ret,
        _toy_markets_config(),
        _toy_costs_config(),
        scenario="zero",
    )

    # Before the signal takes effect, the book is 100% cash (0% return); the day-t jump in A
    # must not leak into the realized return on day t itself.
    assert result.daily_gross_return.loc[signal_date] == 0.0
    # The new weight only takes effect strictly after the signal date.
    execution_date = dates[4]
    assert result.daily_weights.loc[signal_date, "A"] == 0.0
    assert result.daily_weights.loc[execution_date, "A"] == 1.0
