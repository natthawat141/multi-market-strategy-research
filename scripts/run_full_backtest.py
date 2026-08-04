"""Full expanding walk-forward country backtest over every available fold.

Unlike the notebooks (which show one fold in detail for readability), this
script trains tree/MLP/transparent-trend on *every* SPEC.md section 11
walk-forward fold the available real-data history supports, backtests each
fold's out-of-sample test segment independently (no leakage across folds:
each fold's models are trained only on data available before that fold's own
test period), and stitches the consecutive, non-overlapping test segments
into one continuous multi-year out-of-sample equity curve per cost scenario -
this is the closest thing to a "production" result this POC produces.

Usage (from repo root, after `pip install -e ".[data,ml]"`):

    .venv/Scripts/python.exe scripts/run_full_backtest.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from frtbot.backtest.engine import cash_daily_return_series, monthly_rebalance_dates, run_all_cost_scenarios
from frtbot.backtest.splits import generate_walk_forward_folds
from frtbot.config import load_markets_config, load_research_config
from frtbot.data.cache import DataCache
from frtbot.data.fx import align_fx_to_index, fetch_fx_series, identity_fx_series, to_thb_price
from frtbot.data.providers import get_provider
from frtbot.data.quality import detect_extreme_return_days
from frtbot.features.build import MarketSeries, build_country_feature_panel
from frtbot.labels.country import country_classification_label, country_regression_label
from frtbot.models.dataset import build_country_dataset, select_dates
from frtbot.models.ensemble import combine_ensemble
from frtbot.models.mlp import MLPModel
from frtbot.models.tree import TreeModel
from frtbot.models.trend import TransparentTrendModel
from frtbot.portfolio.construction import construct_country_weights
from frtbot.reporting.metrics import summarize_backtest
from frtbot.seed import set_global_seed

REPO_ROOT = Path(__file__).resolve().parent.parent
START = date(2012, 6, 1)
END = date(2026, 7, 31)


def main() -> None:
    set_global_seed(42)
    markets_config = load_markets_config(REPO_ROOT / "configs" / "markets.example.yml")
    research_config = load_research_config(REPO_ROOT / "configs" / "research.yml")
    cache = DataCache(root=REPO_ROOT / "data")
    provider = get_provider("yfinance")

    market_series: dict[str, MarketSeries] = {}
    for m in markets_config.markets:
        df, _ = cache.get_or_fetch(m.key, m.provider_symbol, "ohlcv", provider, START, END)
        anomalies = detect_extreme_return_days(df["close"])
        if len(anomalies) > 0:
            print(
                f"EXCLUDING {m.key} ({m.provider_symbol}): implausible single-day price move(s) "
                f"detected on {[d.date() for d in anomalies]} - likely an upstream data/split "
                f"defect (SPEC.md section 3 data-quality gate), not real price action. "
                f"See frtbot.data.quality for the detection threshold."
            )
            continue
        if m.fx_pair is None:
            fx_aligned = identity_fx_series(df.index)
        else:
            fx_entry = markets_config.fx_for(m)
            rate, _ = fetch_fx_series(fx_entry, cache, provider, START, END)
            fx_aligned = align_fx_to_index(rate, df.index)
        market_series[m.key] = MarketSeries(key=m.key, ohlcv=df, fx_rate_aligned=fx_aligned)

    print(f"Usable markets after data-quality gate: {sorted(market_series.keys())}\n")
    panel = build_country_feature_panel(market_series)
    thb_close = {k: to_thb_price(s.ohlcv["close"], s.fx_rate_aligned) for k, s in market_series.items()}
    thb_daily_return = {k: c.pct_change() for k, c in thb_close.items()}
    reg_labels = {
        k: country_regression_label(c, research_config.cash_annual_rate, research_config.horizon_trading_days)
        for k, c in thb_close.items()
    }
    clf_labels = {k: country_classification_label(v) for k, v in reg_labels.items()}
    long_df = build_country_dataset(panel, reg_labels, clf_labels)

    all_dates = long_df.index.get_level_values("date").unique()
    folds = generate_walk_forward_folds(all_dates, research_config.walk_forward)
    print(f"{len(folds)} walk-forward fold(s) from {START} to {END}")

    vol_63 = {k: panel[k]["vol_63d"] for k in panel}
    scenario_returns: dict[str, list[pd.Series]] = {"zero": [], "optimistic": [], "base": [], "severe": []}

    for fold in folds:
        train_df = select_dates(long_df, fold.train_dates)
        val_df = select_dates(long_df, fold.val_dates)
        test_df = select_dates(long_df, fold.test_dates)

        models = {
            "tree": TreeModel(seed=research_config.seed),
            "mlp": MLPModel(seed=research_config.seed),
            "transparent_trend": TransparentTrendModel(),
        }
        test_scores = {}
        for name, model in models.items():
            model.fit(train_df, val_df)
            test_scores[name] = model.predict(test_df)
        ensemble_result = combine_ensemble(test_scores, research_config.ensemble)
        tree_predicted_return = test_scores["tree"]

        rebalance_dates = monthly_rebalance_dates(fold.test_dates)
        target_weights = {}
        for d in rebalance_dates:
            if d not in ensemble_result.score.index.get_level_values("date"):
                continue
            day_scores = ensemble_result.score.xs(d, level="date")
            day_pred = tree_predicted_return.xs(d, level="date")
            day_vol = pd.Series({k: vol_63[k].get(d, float("nan")) for k in vol_63})
            target_weights[d] = construct_country_weights(day_scores, day_pred, day_vol, research_config.portfolio)

        cash_ret = cash_daily_return_series(fold.test_dates, research_config.cash_annual_rate)
        results = run_all_cost_scenarios(
            fold.test_dates, target_weights, thb_daily_return, cash_ret, markets_config, research_config.costs
        )
        for scenario, result in results.items():
            scenario_returns[scenario].append(result.daily_net_return)

        print(
            f"fold {fold.index}: test {fold.test_dates.min().date()}..{fold.test_dates.max().date()} "
            f"({len(rebalance_dates)} rebalances, {len(target_weights)} with valid signal)"
        )

    print("\nStitched continuous out-of-sample result (all folds' test segments, chronological):")
    rows = []
    for scenario, chunks in scenario_returns.items():
        stitched = pd.concat(chunks).sort_index()
        stitched = stitched[~stitched.index.duplicated(keep="first")]

        class _Result:
            pass

        r = _Result()
        r.scenario = scenario
        r.daily_gross_return = stitched
        r.daily_net_return = stitched
        r.turnover_by_execution_date = pd.Series(dtype=float)
        summary = summarize_backtest(r, risk_free_daily=research_config.cash_annual_rate / 252)
        summary["n_days"] = len(stitched)
        summary["start"] = stitched.index.min().date()
        summary["end"] = stitched.index.max().date()
        rows.append(summary)

    report = pd.DataFrame(rows)[
        ["scenario", "start", "end", "n_days", "cagr_net", "cumulative_return_net",
         "annualized_vol_net", "sharpe_net", "sortino_net", "max_drawdown_net", "recovery_days_net"]
    ]
    pd.set_option("display.width", 160)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
