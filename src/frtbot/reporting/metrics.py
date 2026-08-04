"""Portfolio metrics (SPEC.md section 11 "Primary portfolio metrics")."""

from __future__ import annotations

import numpy as np
import pandas as pd

from frtbot.backtest.engine import BacktestResult

TRADING_DAYS_PER_YEAR = 252


def cagr(daily_return: pd.Series) -> float:
    n_years = len(daily_return) / TRADING_DAYS_PER_YEAR
    if n_years <= 0:
        return float("nan")
    cumulative = float((1.0 + daily_return).prod())
    return cumulative ** (1.0 / n_years) - 1.0


def cumulative_return(daily_return: pd.Series) -> float:
    return float((1.0 + daily_return).prod() - 1.0)


def annualized_vol(daily_return: pd.Series) -> float:
    return float(daily_return.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(daily_return: pd.Series, risk_free_daily: float = 0.0) -> float:
    excess = daily_return - risk_free_daily
    vol = excess.std(ddof=0)
    if vol == 0 or pd.isna(vol):
        return float("nan")
    return float((excess.mean() / vol) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(daily_return: pd.Series, risk_free_daily: float = 0.0) -> float:
    excess = daily_return - risk_free_daily
    downside = excess.clip(upper=0.0)
    downside_vol = downside.std(ddof=0)
    if downside_vol == 0 or pd.isna(downside_vol):
        return float("nan")
    return float((excess.mean() / downside_vol) * np.sqrt(TRADING_DAYS_PER_YEAR))


def equity_curve(daily_return: pd.Series) -> pd.Series:
    return (1.0 + daily_return).cumprod()


def max_drawdown_and_recovery(daily_return: pd.Series) -> dict:
    curve = equity_curve(daily_return)
    running_max = curve.cummax()
    drawdown = curve / running_max - 1.0
    if drawdown.empty:
        return {"max_drawdown": float("nan"), "peak_date": None, "trough_date": None, "recovery_date": None, "recovery_days": None}

    trough_date = drawdown.idxmin()
    max_dd = float(drawdown.min())
    peak_date = curve.loc[:trough_date].idxmax()
    peak_value = curve.loc[peak_date]

    after_trough = curve.loc[trough_date:]
    recovered = after_trough[after_trough >= peak_value]
    recovery_date = recovered.index.min() if len(recovered) else None
    recovery_days = (recovery_date - trough_date).days if recovery_date is not None else None

    return {
        "max_drawdown": max_dd,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "recovery_days": recovery_days,
    }


def turnover_summary(turnover_by_execution_date: pd.Series) -> dict:
    if len(turnover_by_execution_date) == 0:
        return {"total_turnover": 0.0, "avg_turnover_per_rebalance": float("nan"), "n_rebalances": 0}
    return {
        "total_turnover": float(turnover_by_execution_date.sum()),
        "avg_turnover_per_rebalance": float(turnover_by_execution_date.mean()),
        "n_rebalances": int(len(turnover_by_execution_date)),
    }


def cost_drag(gross_return: pd.Series, net_return: pd.Series) -> float:
    return cumulative_return(gross_return) - cumulative_return(net_return)


def worst_period(daily_return: pd.Series, freq: str) -> dict:
    if freq == "month":
        grouped = (1.0 + daily_return).groupby([daily_return.index.year, daily_return.index.month]).prod() - 1.0
    elif freq == "year":
        grouped = (1.0 + daily_return).groupby(daily_return.index.year).prod() - 1.0
    else:
        raise ValueError(f"Unknown freq {freq!r}")
    if grouped.empty:
        return {f"worst_{freq}_return": float("nan"), f"worst_{freq}_period": None}
    return {f"worst_{freq}_return": float(grouped.min()), f"worst_{freq}_period": grouped.idxmin()}


def country_attribution(
    daily_weights: pd.DataFrame, daily_thb_return_by_market: dict[str, pd.Series]
) -> pd.Series:
    """Approximate gross return contribution by market: sum_t weight_{t,m} * return_{t,m}."""
    contributions = {}
    for m, ret in daily_thb_return_by_market.items():
        if m not in daily_weights.columns:
            continue
        aligned = ret.reindex(daily_weights.index).fillna(0.0)
        contributions[m] = float((daily_weights[m] * aligned).sum())
    return pd.Series(contributions).sort_values(ascending=False)


def summarize_backtest(result: BacktestResult, risk_free_daily: float = 0.0) -> dict:
    gross, net = result.daily_gross_return, result.daily_net_return
    mdd = max_drawdown_and_recovery(net)
    return {
        "scenario": result.scenario,
        "cagr_gross": cagr(gross),
        "cagr_net": cagr(net),
        "cumulative_return_gross": cumulative_return(gross),
        "cumulative_return_net": cumulative_return(net),
        "annualized_vol_net": annualized_vol(net),
        "sharpe_net": sharpe_ratio(net, risk_free_daily),
        "sortino_net": sortino_ratio(net, risk_free_daily),
        "max_drawdown_net": mdd["max_drawdown"],
        "recovery_days_net": mdd["recovery_days"],
        "cost_drag": cost_drag(gross, net),
        **turnover_summary(result.turnover_by_execution_date),
        **worst_period(net, "month"),
        **worst_period(net, "year"),
    }


def summarize_all_scenarios(results: dict[str, BacktestResult], risk_free_daily: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame([summarize_backtest(r, risk_free_daily) for r in results.values()])
