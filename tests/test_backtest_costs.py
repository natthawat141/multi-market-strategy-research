from __future__ import annotations

import pandas as pd
import pytest

from frtbot.backtest.engine import cash_daily_return_series, run_all_cost_scenarios, run_backtest
from frtbot.config import CostsConfig, MarketEntry, MarketsConfig


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


def _flat_return_dates(n: int = 20) -> tuple[pd.DatetimeIndex, dict[str, pd.Series]]:
    dates = pd.bdate_range("2020-01-01", periods=n)
    zero_ret = pd.Series(0.0, index=dates)
    return dates, {"A": zero_ret.copy(), "B": zero_ret.copy()}


def test_zero_scenario_has_zero_cost():
    dates, returns = _flat_return_dates()
    signal_date = dates[2]
    weights = {signal_date: pd.Series({"A": 1.0})}
    cash_ret = cash_daily_return_series(dates, 0.0)

    result = run_backtest(dates, weights, returns, cash_ret, _toy_markets_config(), _toy_costs_config(), "zero")
    assert (result.cost_by_execution_date == 0.0).all()
    pd.testing.assert_series_equal(result.daily_gross_return, result.daily_net_return, check_names=False)


def test_cost_scales_with_turnover_and_bps():
    dates, returns = _flat_return_dates()
    signal_date = dates[2]
    weights = {signal_date: pd.Series({"A": 1.0})}  # turnover = |1.0 - 0.0| = 1.0 on A
    cash_ret = cash_daily_return_series(dates, 0.0)
    costs_cfg = _toy_costs_config()

    result = run_backtest(dates, weights, returns, cash_ret, _toy_markets_config(), costs_cfg, "base")
    execution_date = dates[3]
    expected_cost = 1.0 * costs_cfg.bps_for("country_proxy", "base") / 10000.0
    assert result.cost_by_execution_date.loc[execution_date] == pytest.approx(expected_cost)
    assert result.turnover_by_execution_date.loc[execution_date] == pytest.approx(1.0)


def test_net_return_equals_gross_minus_cost():
    dates, returns = _flat_return_dates()
    returns["A"].iloc[3:] = 0.01  # steady 1%/day once held
    signal_date = dates[2]
    weights = {signal_date: pd.Series({"A": 1.0})}
    cash_ret = cash_daily_return_series(dates, 0.0)

    result = run_backtest(dates, weights, returns, cash_ret, _toy_markets_config(), _toy_costs_config(), "base")
    pd.testing.assert_series_equal(
        result.daily_net_return, result.daily_gross_return - result.cost_by_execution_date.reindex(dates).fillna(0.0),
        check_names=False,
    )


def test_more_severe_scenarios_never_produce_higher_net_return():
    dates, returns = _flat_return_dates()
    returns["A"].iloc[3:] = 0.001
    signal_date = dates[2]
    weights = {signal_date: pd.Series({"A": 1.0})}
    cash_ret = cash_daily_return_series(dates, 0.0)

    results = run_all_cost_scenarios(dates, weights, returns, cash_ret, _toy_markets_config(), _toy_costs_config())
    cum_by_scenario = {
        name: float((1 + r.daily_net_return).prod()) for name, r in results.items()
    }
    ordered = [cum_by_scenario[s] for s in ("zero", "optimistic", "base", "severe")]
    assert ordered == sorted(ordered, reverse=True)


def test_rebalancing_back_to_same_weights_still_charges_round_trip_cost():
    """SPEC.md: costs are charged on absolute turnover - selling then re-buying is not free."""
    dates, returns = _flat_return_dates()
    d0, d1 = dates[2], dates[6]
    weights = {d0: pd.Series({"A": 1.0}), d1: pd.Series({"B": 1.0})}
    cash_ret = cash_daily_return_series(dates, 0.0)
    costs_cfg = _toy_costs_config()

    result = run_backtest(dates, weights, returns, cash_ret, _toy_markets_config(), costs_cfg, "base")
    bps = costs_cfg.bps_for("country_proxy", "base") / 10000.0
    exec1, exec2 = dates[3], dates[7]
    assert result.cost_by_execution_date.loc[exec1] == pytest.approx(1.0 * bps)  # buy A
    assert result.cost_by_execution_date.loc[exec2] == pytest.approx(2.0 * bps)  # sell A, buy B
