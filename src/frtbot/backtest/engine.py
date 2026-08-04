"""Monthly-rebalance backtest engine (SPEC.md section 10).

Execution model and its documented simplification: a signal computed from
data through the close of rebalance date `t` selects target weights that
become effective starting at the next tradable date `t+1` (never at `t`'s own
close-to-close return, satisfying "never trade at the same close used to
calculate a signal"). Full open-vs-close intraday accounting for `t+1`
(SPEC.md 10: "execute at the next tradable local-market open when open prices
exist") is deferred - this POC applies the new weights to `t+1`'s full
close-to-close return, a standard simplification for a daily-bar, monthly-
rebalance backtest. This is documented in README's limitations section.

Eligibility gate vs. ranking (SPEC.md 9.1) - "positive predicted THB excess
return" uses the tree model's native regression prediction (its output is
literally in `label_regression` units); the frozen ensemble rank score
(SPEC.md 8.4) is used only to rank/select among eligible markets. This
resolves an ambiguity SPEC.md leaves to the implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from frtbot.config import CostScenario, CostsConfig, MarketsConfig

TRADING_DAYS_PER_YEAR = 252


def cash_daily_return_series(dates: pd.DatetimeIndex, cash_annual_rate: float) -> pd.Series:
    daily_rate = (1.0 + cash_annual_rate) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    return pd.Series(daily_rate, index=dates, name="cash_daily_return")


def monthly_rebalance_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading date of each calendar month present in `dates` (signal dates)."""
    dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    s = pd.Series(dates, index=dates)
    last_per_month = s.groupby([dates.year, dates.month]).max()
    return pd.DatetimeIndex(sorted(last_per_month.values))


@dataclass
class BacktestResult:
    scenario: CostScenario
    daily_gross_return: pd.Series
    daily_net_return: pd.Series
    daily_weights: pd.DataFrame
    turnover_by_execution_date: pd.Series
    cost_by_execution_date: pd.Series


def run_backtest(
    dates: pd.DatetimeIndex,
    target_weights_by_signal_date: dict[pd.Timestamp, pd.Series],
    daily_thb_return_by_market: dict[str, pd.Series],
    cash_daily_return: pd.Series,
    markets_config: MarketsConfig,
    costs_config: CostsConfig,
    scenario: CostScenario,
    cash_key: str = "CASH_THB",
) -> BacktestResult:
    dates = pd.DatetimeIndex(sorted(dates))

    execution_date_for_signal: dict[pd.Timestamp, pd.Timestamp] = {}
    for sd in sorted(target_weights_by_signal_date):
        pos = dates.searchsorted(sd, side="right")
        if pos < len(dates):
            execution_date_for_signal[dates[pos]] = sd

    current_weights = pd.Series({cash_key: 1.0})
    gross_returns: dict[pd.Timestamp, float] = {}
    net_costs: dict[pd.Timestamp, float] = {}
    turnovers: dict[pd.Timestamp, float] = {}
    weight_rows: dict[pd.Timestamp, pd.Series] = {}

    for d in dates:
        cost = 0.0
        if d in execution_date_for_signal:
            new_weights = target_weights_by_signal_date[execution_date_for_signal[d]]
            names = (set(current_weights.index) | set(new_weights.index)) - {cash_key}
            turnover = 0.0
            for n in names:
                delta = abs(float(new_weights.get(n, 0.0)) - float(current_weights.get(n, 0.0)))
                turnover += delta
                instrument_class = markets_config.by_key(n).instrument_class
                bps = costs_config.bps_for(instrument_class, scenario)
                cost += delta * bps / 10000.0
            turnovers[d] = turnover
            current_weights = new_weights

        day_ret = 0.0
        for n, w in current_weights.items():
            if w == 0.0:
                continue
            r = cash_daily_return.get(d, 0.0) if n == cash_key else daily_thb_return_by_market[n].get(d, 0.0)
            if pd.isna(r):
                r = 0.0
            day_ret += float(w) * float(r)

        gross_returns[d] = day_ret
        net_costs[d] = cost
        weight_rows[d] = current_weights.copy()

    gross = pd.Series(gross_returns).reindex(dates).fillna(0.0)
    cost_series = pd.Series(net_costs).reindex(dates).fillna(0.0)
    net = gross - cost_series
    weights_df = pd.DataFrame(weight_rows).T.reindex(dates).fillna(0.0)
    turnover_series = pd.Series(turnovers).sort_index()

    return BacktestResult(
        scenario=scenario,
        daily_gross_return=gross,
        daily_net_return=net,
        daily_weights=weights_df,
        turnover_by_execution_date=turnover_series,
        cost_by_execution_date=cost_series.loc[turnover_series.index] if len(turnover_series) else cost_series.iloc[0:0],
    )


def run_all_cost_scenarios(
    dates: pd.DatetimeIndex,
    target_weights_by_signal_date: dict[pd.Timestamp, pd.Series],
    daily_thb_return_by_market: dict[str, pd.Series],
    cash_daily_return: pd.Series,
    markets_config: MarketsConfig,
    costs_config: CostsConfig,
    cash_key: str = "CASH_THB",
) -> dict[CostScenario, BacktestResult]:
    scenarios: tuple[CostScenario, ...] = ("zero", "optimistic", "base", "severe")
    return {
        scenario: run_backtest(
            dates,
            target_weights_by_signal_date,
            daily_thb_return_by_market,
            cash_daily_return,
            markets_config,
            costs_config,
            scenario,
            cash_key,
        )
        for scenario in scenarios
    }
