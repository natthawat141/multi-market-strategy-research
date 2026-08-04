"""Export reproducible, local-only datasets for the detailed FRTBOT report.

The script deliberately re-runs every expanding walk-forward fold so the
report is backed by inspectable rows rather than values copied from README.
It writes only to the gitignored ``artifacts/detailed_report/data`` folder.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from frtbot.backtest.engine import (
    BacktestResult,
    cash_daily_return_series,
    monthly_rebalance_dates,
    run_all_cost_scenarios,
    run_backtest,
)
from frtbot.backtest.splits import generate_walk_forward_folds
from frtbot.config import load_markets_config, load_research_config
from frtbot.data.cache import DataCache
from frtbot.data.fx import align_fx_to_index, fetch_fx_series, identity_fx_series, to_thb_price
from frtbot.data.providers import get_provider
from frtbot.data.quality import detect_extreme_return_days
from frtbot.features.build import MarketSeries, build_country_feature_panel
from frtbot.labels.country import country_classification_label, country_regression_label
from frtbot.models.baselines import EqualWeightModel, RidgeBaselineModel, RuleMomentumModel
from frtbot.models.dataset import build_country_dataset, feature_columns, select_dates
from frtbot.models.ensemble import combine_ensemble
from frtbot.models.mlp import MLPModel
from frtbot.models.tree import TreeModel
from frtbot.models.trend import TransparentTrendModel
from frtbot.portfolio.construction import construct_country_weights
from frtbot.reporting.data_audit import build_data_audit
from frtbot.reporting.metrics import (
    country_attribution,
    equity_curve,
    summarize_backtest,
)
from frtbot.reporting.ml_metrics import rank_information_coefficient, top_quintile_hit_rate
from frtbot.seed import set_global_seed

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "artifacts" / "detailed_report" / "data"
START = date(2012, 6, 1)
END = date(2026, 7, 31)
AS_OF = pd.Timestamp("2026-08-04")


def _stitch(chunks: list[pd.Series]) -> pd.Series:
    series = pd.concat(chunks).sort_index()
    return series[~series.index.duplicated(keep="first")]


def _stitch_frame(chunks: list[pd.DataFrame]) -> pd.DataFrame:
    frame = pd.concat(chunks).sort_index()
    return frame[~frame.index.duplicated(keep="first")]


def _result_from_series(name: str, net: pd.Series, gross: pd.Series | None = None) -> BacktestResult:
    gross = net if gross is None else gross
    return BacktestResult(
        scenario=name,
        daily_gross_return=gross,
        daily_net_return=net,
        daily_weights=pd.DataFrame(index=net.index),
        turnover_by_execution_date=pd.Series(dtype=float),
        cost_by_execution_date=pd.Series(dtype=float),
    )


def _summary_row(result: BacktestResult, risk_free_daily: float, category: str) -> dict:
    row = summarize_backtest(result, risk_free_daily=risk_free_daily)
    row.update(
        {
            "series": result.scenario,
            "category": category,
            "start": result.daily_net_return.index.min().date(),
            "end": result.daily_net_return.index.max().date(),
            "n_days": len(result.daily_net_return),
        }
    )
    return row


def _safe_metric(metric, score: pd.Series, label: pd.Series) -> float:
    with np.errstate(all="ignore"):
        return float(metric(score, label))


def _monthly_compound(series: pd.Series) -> pd.Series:
    return (1.0 + series).groupby(series.index.to_period("M")).prod() - 1.0


def _annual_compound(series: pd.Series) -> pd.Series:
    return (1.0 + series).groupby(series.index.year).prod() - 1.0


def main() -> None:
    set_global_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    markets_config = load_markets_config(REPO_ROOT / "configs" / "markets.example.yml")
    research_config = load_research_config(REPO_ROOT / "configs" / "research.yml")
    cache = DataCache(root=REPO_ROOT / "data")
    provider = get_provider("yfinance")

    audit = build_data_audit(markets_config, cache, as_of=AS_OF)
    audit.to_csv(OUTPUT_DIR / "data_audit.csv", index=False)

    anomaly_rows: list[dict] = []
    market_series: dict[str, MarketSeries] = {}
    for market in markets_config.markets:
        frame, _ = cache.get_or_fetch(
            market.key, market.provider_symbol, "ohlcv", provider, START, END
        )
        anomalies = detect_extreme_return_days(frame["close"])
        if len(anomalies):
            for anomaly_date in anomalies:
                anomaly_rows.append(
                    {
                        "market": market.key,
                        "provider_symbol": market.provider_symbol,
                        "date": anomaly_date.date(),
                        "daily_return": float(frame["close"].pct_change().loc[anomaly_date]),
                        "action": "excluded_from_model_and_backtest",
                    }
                )
            continue

        if market.fx_pair is None:
            fx_aligned = identity_fx_series(frame.index)
        else:
            fx_entry = markets_config.fx_for(market)
            fx_rate, _ = fetch_fx_series(fx_entry, cache, provider, START, END)
            fx_aligned = align_fx_to_index(fx_rate, frame.index)
        market_series[market.key] = MarketSeries(
            key=market.key, ohlcv=frame, fx_rate_aligned=fx_aligned
        )

    pd.DataFrame(anomaly_rows).to_csv(OUTPUT_DIR / "price_anomalies.csv", index=False)

    panel = build_country_feature_panel(market_series)
    thb_close = {
        key: to_thb_price(series.ohlcv["close"], series.fx_rate_aligned)
        for key, series in market_series.items()
    }
    thb_daily_return = {key: close.pct_change() for key, close in thb_close.items()}
    reg_labels = {
        key: country_regression_label(
            close,
            research_config.cash_annual_rate,
            research_config.horizon_trading_days,
        )
        for key, close in thb_close.items()
    }
    clf_labels = {key: country_classification_label(label) for key, label in reg_labels.items()}
    long_df = build_country_dataset(panel, reg_labels, clf_labels)
    long_df.to_parquet(OUTPUT_DIR / "model_panel.parquet")

    feature_catalog = pd.DataFrame(
        {
            "feature": feature_columns(long_df),
            "dtype": [str(long_df[column].dtype) for column in feature_columns(long_df)],
            "non_null_rows": [int(long_df[column].notna().sum()) for column in feature_columns(long_df)],
            "null_rate": [float(long_df[column].isna().mean()) for column in feature_columns(long_df)],
        }
    )
    feature_catalog.to_csv(OUTPUT_DIR / "feature_catalog.csv", index=False)

    all_dates = long_df.index.get_level_values("date").unique()
    folds = generate_walk_forward_folds(all_dates, research_config.walk_forward)
    vol_63 = {key: panel[key]["vol_63d"] for key in panel}
    risk_free_daily = research_config.cash_annual_rate / 252

    scenario_names = ("zero", "optimistic", "base", "severe")
    scenario_net: dict[str, list[pd.Series]] = {name: [] for name in scenario_names}
    scenario_gross: dict[str, list[pd.Series]] = {name: [] for name in scenario_names}
    scenario_turnover: dict[str, list[pd.Series]] = {name: [] for name in scenario_names}
    base_weight_chunks: list[pd.DataFrame] = []
    rebalance_weight_rows: list[dict] = []
    fold_rows: list[dict] = []
    model_metric_rows: list[dict] = []

    benchmark_names = ["equal_weight"] + [f"buy_hold_{key}" for key in sorted(market_series)] + ["cash"]
    benchmark_net: dict[str, list[pd.Series]] = {name: [] for name in benchmark_names}
    benchmark_gross: dict[str, list[pd.Series]] = {name: [] for name in benchmark_names}
    benchmark_turnover: dict[str, list[pd.Series]] = {name: [] for name in benchmark_names}

    for fold in folds:
        train_df = select_dates(long_df, fold.train_dates)
        val_df = select_dates(long_df, fold.val_dates)
        test_df = select_dates(long_df, fold.test_dates)

        models = {
            "tree": TreeModel(seed=research_config.seed),
            "mlp": MLPModel(seed=research_config.seed),
            "transparent_trend": TransparentTrendModel(),
            "linear_ridge": RidgeBaselineModel(seed=research_config.seed),
            "rule_momentum": RuleMomentumModel(),
            "equal_weight_score": EqualWeightModel(),
        }
        scores: dict[str, pd.Series] = {}
        for name, model in models.items():
            model.fit(train_df, val_df)
            scores[name] = model.predict(test_df)

        ensemble_result = combine_ensemble(
            {name: scores[name] for name in ("tree", "mlp", "transparent_trend")},
            research_config.ensemble,
        )
        scores["ensemble"] = ensemble_result.score
        label = test_df["label_regression"]
        for model_name, score in scores.items():
            model_metric_rows.append(
                {
                    "fold": fold.index,
                    "model": model_name,
                    "test_start": fold.test_dates.min().date(),
                    "test_end": fold.test_dates.max().date(),
                    "n_scored_rows": int(pd.DataFrame({"score": score, "label": label}).dropna().shape[0]),
                    "rank_ic": _safe_metric(rank_information_coefficient, score, label),
                    "top_quintile_hit_rate": _safe_metric(top_quintile_hit_rate, score, label),
                }
            )

        rebalance_dates = monthly_rebalance_dates(fold.test_dates)
        target_weights: dict[pd.Timestamp, pd.Series] = {}
        for signal_date in rebalance_dates:
            if signal_date not in ensemble_result.score.index.get_level_values("date"):
                continue
            day_scores = ensemble_result.score.xs(signal_date, level="date")
            day_prediction = scores["tree"].xs(signal_date, level="date")
            day_vol = pd.Series(
                {key: vol_63[key].get(signal_date, float("nan")) for key in vol_63}
            )
            weights = construct_country_weights(
                day_scores, day_prediction, day_vol, research_config.portfolio
            )
            target_weights[signal_date] = weights
            for asset, weight in weights.items():
                rebalance_weight_rows.append(
                    {
                        "fold": fold.index,
                        "signal_date": signal_date.date(),
                        "asset": asset,
                        "target_weight": float(weight),
                    }
                )

        cash_return = cash_daily_return_series(fold.test_dates, research_config.cash_annual_rate)
        results = run_all_cost_scenarios(
            fold.test_dates,
            target_weights,
            thb_daily_return,
            cash_return,
            markets_config,
            research_config.costs,
        )
        for scenario, result in results.items():
            scenario_net[scenario].append(result.daily_net_return)
            scenario_gross[scenario].append(result.daily_gross_return)
            scenario_turnover[scenario].append(result.turnover_by_execution_date)
        base_weight_chunks.append(results["base"].daily_weights)

        equal_weights = pd.Series(
            {key: 1.0 / len(market_series) for key in sorted(market_series)}
        )
        equal_targets = {signal_date: equal_weights for signal_date in rebalance_dates}
        equal_result = run_backtest(
            fold.test_dates,
            equal_targets,
            thb_daily_return,
            cash_return,
            markets_config,
            research_config.costs,
            "base",
        )
        benchmark_net["equal_weight"].append(equal_result.daily_net_return)
        benchmark_gross["equal_weight"].append(equal_result.daily_gross_return)
        benchmark_turnover["equal_weight"].append(equal_result.turnover_by_execution_date)

        fold_summary = summarize_backtest(results["base"], risk_free_daily=risk_free_daily)
        equal_summary = summarize_backtest(equal_result, risk_free_daily=risk_free_daily)
        fold_summary.update(
            {
                "fold": fold.index,
                "test_start": fold.test_dates.min().date(),
                "test_end": fold.test_dates.max().date(),
                "train_rows": len(train_df),
                "validation_rows": len(val_df),
                "test_rows": len(test_df),
                "rebalance_signals": len(target_weights),
                "equal_weight_cagr_net": equal_summary["cagr_net"],
                "equal_weight_cumulative_return_net": equal_summary["cumulative_return_net"],
                "equal_weight_sharpe_net": equal_summary["sharpe_net"],
                "cagr_spread_vs_equal_weight": fold_summary["cagr_net"] - equal_summary["cagr_net"],
            }
        )
        fold_rows.append(fold_summary)

        for key in sorted(market_series):
            targets = {rebalance_dates[0]: pd.Series({key: 1.0})}
            buy_hold = run_backtest(
                fold.test_dates,
                targets,
                thb_daily_return,
                cash_return,
                markets_config,
                research_config.costs,
                "base",
            )
            benchmark_net[f"buy_hold_{key}"].append(buy_hold.daily_net_return)
            benchmark_gross[f"buy_hold_{key}"].append(buy_hold.daily_gross_return)
            benchmark_turnover[f"buy_hold_{key}"].append(buy_hold.turnover_by_execution_date)

        benchmark_net["cash"].append(cash_return)
        benchmark_gross["cash"].append(cash_return)
        benchmark_turnover["cash"].append(pd.Series(dtype=float))

    scenario_results: dict[str, BacktestResult] = {}
    scenario_summary_rows: list[dict] = []
    for scenario in scenario_names:
        net = _stitch(scenario_net[scenario])
        gross = _stitch(scenario_gross[scenario])
        turnover = _stitch(scenario_turnover[scenario])
        result = BacktestResult(
            scenario=scenario,
            daily_gross_return=gross,
            daily_net_return=net,
            daily_weights=pd.DataFrame(index=net.index),
            turnover_by_execution_date=turnover,
            cost_by_execution_date=pd.Series(dtype=float),
        )
        scenario_results[scenario] = result
        scenario_summary_rows.append(_summary_row(result, risk_free_daily, "strategy_cost_scenario"))

    benchmark_results: dict[str, BacktestResult] = {}
    benchmark_summary_rows: list[dict] = []
    for name in benchmark_names:
        net = _stitch(benchmark_net[name])
        gross = _stitch(benchmark_gross[name])
        turnover = _stitch(benchmark_turnover[name])
        result = BacktestResult(
            scenario=name,
            daily_gross_return=gross,
            daily_net_return=net,
            daily_weights=pd.DataFrame(index=net.index),
            turnover_by_execution_date=turnover,
            cost_by_execution_date=pd.Series(dtype=float),
        )
        benchmark_results[name] = result
        benchmark_summary_rows.append(_summary_row(result, risk_free_daily, "benchmark"))

    pd.DataFrame(scenario_summary_rows).to_csv(OUTPUT_DIR / "scenario_summary.csv", index=False)
    pd.DataFrame(benchmark_summary_rows).to_csv(OUTPUT_DIR / "benchmark_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUTPUT_DIR / "fold_summary.csv", index=False)
    pd.DataFrame(model_metric_rows).to_csv(OUTPUT_DIR / "model_metrics_by_fold.csv", index=False)
    pd.DataFrame(rebalance_weight_rows).to_csv(OUTPUT_DIR / "rebalance_weights.csv", index=False)

    daily = pd.DataFrame(
        {f"strategy_{name}_return": result.daily_net_return for name, result in scenario_results.items()}
        | {f"benchmark_{name}_return": result.daily_net_return for name, result in benchmark_results.items()}
    ).sort_index()
    for name, result in scenario_results.items():
        daily[f"strategy_{name}_equity"] = equity_curve(result.daily_net_return)
        curve = daily[f"strategy_{name}_equity"]
        daily[f"strategy_{name}_drawdown"] = curve / curve.cummax() - 1.0
    for name, result in benchmark_results.items():
        daily[f"benchmark_{name}_equity"] = equity_curve(result.daily_net_return)
    daily.index.name = "date"
    daily.to_csv(OUTPUT_DIR / "daily_performance.csv")

    monthly_columns = {}
    annual_columns = {}
    for column in [c for c in daily.columns if c.endswith("_return")]:
        monthly_columns[column] = _monthly_compound(daily[column])
        annual_columns[column] = _annual_compound(daily[column])
    monthly = pd.DataFrame(monthly_columns)
    monthly.index = monthly.index.astype(str)
    monthly.index.name = "month"
    monthly.to_csv(OUTPUT_DIR / "monthly_returns.csv")
    annual = pd.DataFrame(annual_columns)
    annual.index.name = "year"
    annual.to_csv(OUTPUT_DIR / "annual_returns.csv")

    base_weights = _stitch_frame(base_weight_chunks)
    base_weights.index.name = "date"
    base_weights.to_csv(OUTPUT_DIR / "daily_weights.csv")
    attribution = country_attribution(base_weights, thb_daily_return).rename("gross_return_contribution")
    attribution.rename_axis("market").reset_index().to_csv(
        OUTPUT_DIR / "country_attribution.csv", index=False
    )

    oos_dates = daily.index
    market_rows = []
    for key, returns in thb_daily_return.items():
        oos = returns.reindex(oos_dates).fillna(0.0)
        result = _result_from_series(key, oos)
        summary = _summary_row(result, risk_free_daily, "market_diagnostic")
        summary["market"] = key
        market_rows.append(summary)
    pd.DataFrame(market_rows).to_csv(OUTPUT_DIR / "market_oos_summary.csv", index=False)

    run_manifest = pd.DataFrame(
        [
            {"key": "generated_at", "value": AS_OF.isoformat()},
            {"key": "data_start", "value": str(START)},
            {"key": "data_end_exclusive", "value": str(END)},
            {"key": "usable_markets", "value": ",".join(sorted(market_series))},
            {"key": "excluded_markets", "value": ",".join(sorted({row['market'] for row in anomaly_rows}))},
            {"key": "walk_forward_folds", "value": str(len(folds))},
            {"key": "base_currency", "value": research_config.base_currency},
            {"key": "horizon_trading_days", "value": str(research_config.horizon_trading_days)},
            {"key": "seed", "value": str(research_config.seed)},
        ]
    )
    run_manifest.to_csv(OUTPUT_DIR / "run_manifest.csv", index=False)

    print(f"Detailed report data written to {OUTPUT_DIR}")
    print(f"Usable markets: {sorted(market_series)}; excluded: {sorted({row['market'] for row in anomaly_rows})}")
    print(pd.DataFrame(scenario_summary_rows)[["series", "cagr_net", "cumulative_return_net", "sharpe_net", "max_drawdown_net"]].to_string(index=False))
    print("\nBenchmarks:")
    print(pd.DataFrame(benchmark_summary_rows)[["series", "cagr_net", "cumulative_return_net", "sharpe_net", "max_drawdown_net"]].to_string(index=False))


if __name__ == "__main__":
    main()
