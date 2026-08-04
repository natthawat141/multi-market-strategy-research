"""Generator for notebooks/*.ipynb - not itself a reusable package module.

Builds each notebook's cells with nbformat, executes it against a registered
`frtbot-venv` Jupyter kernel via nbclient (downloading/caching real market
data as a side effect into the local, gitignored `data/` directory), and
writes the executed notebook (with outputs) to notebooks/. Re-run whenever
the notebook cells or underlying pipeline change:

    .venv/Scripts/python.exe -m ipykernel install --user --name frtbot-venv \\
        --display-name "Python 3 (frtbot .venv)"   # once per machine
    .venv/Scripts/python.exe scripts/build_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT / "notebooks"


def md(src: str):
    return new_markdown_cell(src.strip())


def code(src: str):
    return new_code_cell(src.strip())


# ---------------------------------------------------------------------------
# 00_data_audit.ipynb
# ---------------------------------------------------------------------------

CELLS_00 = [
    md(
        """
# 00 - Data Audit

FRTBOT research POC — country-level vertical slice (SPEC.md M1/M2).

This notebook downloads a bounded real-data sample for all five configured
market proxies and FX pairs, then runs the SPEC.md-required data audit
(coverage, gaps, staleness, currency, adjustment metadata; usable / proxy /
disabled / missing / stale / synthetic classification).

**Data is REAL** (Yahoo Finance via `yfinance`), not `SYNTHETIC` — see the
`source` column below. Retrieved on the date this notebook was last executed.
"""
    ),
    code(
        """
from datetime import date
from pathlib import Path

import pandas as pd

from frtbot.config import load_markets_config
from frtbot.data.cache import DataCache
from frtbot.data.providers import get_provider
from frtbot.reporting.data_audit import build_data_audit

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

START = date(2012, 6, 1)
END = date(2026, 7, 31)  # bounded sample; see README for the deferred full-history command

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "configs").exists():
    REPO_ROOT = REPO_ROOT.parent

markets_config = load_markets_config(REPO_ROOT / "configs" / "markets.example.yml")
cache = DataCache(root=REPO_ROOT / "data")
provider = get_provider("yfinance")

print(f"Base currency: {markets_config.base_currency}")
print(f"Markets: {[m.key for m in markets_config.markets]}")
"""
    ),
    code(
        """
for m in markets_config.markets:
    if m.mode == "disabled":
        print(f"{m.key}: disabled ({m.disabled_reason}) - skipping download")
        continue
    df, meta = cache.get_or_fetch(m.key, m.provider_symbol, "ohlcv", provider, START, END)
    print(f"{m.key:4s} {m.provider_symbol:12s} rows={len(df):5d} "
          f"{df.index.min().date()} .. {df.index.max().date()} source={meta['source']}")

from frtbot.data.fx import fetch_fx_series

for fx in markets_config.fx:
    rate, meta = fetch_fx_series(fx, cache, provider, START, END)
    print(f"{fx.pair:8s} {fx.provider_symbol:10s} rows={len(rate):5d} "
          f"{rate.index.min().date()} .. {rate.index.max().date()} "
          f"invert={fx.invert} source={meta['source']}")
"""
    ),
    md("## Data audit report"),
    code(
        """
audit = build_data_audit(markets_config, cache, as_of=pd.Timestamp(END))
display_cols = [
    "key", "kind", "status", "source", "currency", "provider", "provider_symbol",
    "adjustment_type", "first_valid_date", "last_valid_date", "row_count",
    "business_day_coverage_ratio", "max_gap_days", "price_anomaly_dates",
]
audit[display_cols]
"""
    ),
    code(
        """
print("Status counts:")
print(audit["status"].value_counts())
assert (audit["status"].isin(["proxy", "usable"])).all(), "Unexpected non-usable series in the audit"
assert (audit["source"] == "REAL").all(), "Expected only REAL data in this notebook run"
print("\\nAll five markets + four FX pairs are REAL and usable/proxy - no disabled/missing/stale series.")

anomalous = audit[audit["price_anomaly_dates"].apply(len) > 0]
if len(anomalous):
    print("\\nWARNING: implausible single-day price move(s) detected (likely an upstream "
          "provider/split defect, not real price action) - SPEC.md section 3 data-quality gate:")
    print(anomalous[["key", "provider_symbol", "price_anomaly_dates"]].to_string(index=False))
    print("\\nSee frtbot.data.quality.detect_extreme_return_days. These markets are excluded "
          "from the modeling/backtest notebooks (01, 02) and scripts/run_full_backtest.py for "
          "any affected run - see their 'data-quality gate' cell.")
"""
    ),
]


# ---------------------------------------------------------------------------
# 01_global_country_model.ipynb
# ---------------------------------------------------------------------------

CELLS_01 = [
    md(
        """
# 01 - Global Country Model

Builds the leakage-safe feature/label panel for all five markets (SPEC.md
section 6-7), runs one expanding walk-forward fold (SPEC.md section 11),
trains every required baseline plus the tree/MLP/transparent-trend models
(SPEC.md section 8), and compares them **on identical out-of-sample dates**
using rank information coefficient and top-quintile hit rate.

Reuses the same local REAL data cache populated by `00_data_audit.ipynb`
(re-running this notebook standalone will download it if missing).
"""
    ),
    code(
        """
from datetime import date
from pathlib import Path

import pandas as pd

from frtbot.config import load_markets_config, load_research_config
from frtbot.data.cache import DataCache
from frtbot.data.providers import get_provider
from frtbot.data.fx import align_fx_to_index, fetch_fx_series, identity_fx_series, to_thb_price
from frtbot.data.quality import detect_extreme_return_days
from frtbot.features.build import MarketSeries, build_country_feature_panel
from frtbot.labels.country import country_classification_label, country_regression_label
from frtbot.models.dataset import build_country_dataset, select_dates
from frtbot.backtest.splits import generate_walk_forward_folds
from frtbot.models.baselines import EqualWeightModel, RidgeBaselineModel, RuleMomentumModel
from frtbot.models.tree import TreeModel
from frtbot.models.mlp import MLPModel
from frtbot.models.trend import TransparentTrendModel
from frtbot.models.ensemble import combine_ensemble
from frtbot.reporting.ml_metrics import rank_information_coefficient, top_quintile_hit_rate
from frtbot.seed import set_global_seed

pd.set_option("display.width", 140)
set_global_seed(42)

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "configs").exists():
    REPO_ROOT = REPO_ROOT.parent

START = date(2012, 6, 1)
END = date(2026, 7, 31)

markets_config = load_markets_config(REPO_ROOT / "configs" / "markets.example.yml")
research_config = load_research_config(REPO_ROOT / "configs" / "research.yml")
cache = DataCache(root=REPO_ROOT / "data")
provider = get_provider("yfinance")
"""
    ),
    md("## Build features, labels, and the long-format modeling panel"),
    code(
        """
market_series: dict[str, MarketSeries] = {}
for m in markets_config.markets:
    df, _ = cache.get_or_fetch(m.key, m.provider_symbol, "ohlcv", provider, START, END)

    # Data-quality gate (SPEC.md section 3): exclude markets with an implausible single-day
    # move - see notebook 00's audit, which flags this same check. Discovered in practice:
    # 1306.T (JP) has a two-day misadjustment (likely an unadjusted stock split) in the
    # upstream Yahoo Finance feed; including it corrupts volatility/return calculations.
    anomalies = detect_extreme_return_days(df["close"])
    if len(anomalies) > 0:
        print(f"EXCLUDING {m.key} ({m.provider_symbol}): price anomaly on {[d.date() for d in anomalies]}")
        continue

    if m.fx_pair is None:
        fx_aligned = identity_fx_series(df.index)
    else:
        fx_entry = markets_config.fx_for(m)
        rate, _ = fetch_fx_series(fx_entry, cache, provider, START, END)
        fx_aligned = align_fx_to_index(rate, df.index)
    market_series[m.key] = MarketSeries(key=m.key, ohlcv=df, fx_rate_aligned=fx_aligned)

print(f"Usable markets after data-quality gate: {sorted(market_series.keys())}")

panel = build_country_feature_panel(market_series)
thb_close = {k: to_thb_price(s.ohlcv["close"], s.fx_rate_aligned) for k, s in market_series.items()}
thb_daily_return = {k: c.pct_change() for k, c in thb_close.items()}

reg_labels = {
    k: country_regression_label(c, research_config.cash_annual_rate, research_config.horizon_trading_days)
    for k, c in thb_close.items()
}
clf_labels = {k: country_classification_label(v) for k, v in reg_labels.items()}

long_df = build_country_dataset(panel, reg_labels, clf_labels)
print("Long-format panel shape:", long_df.shape)
long_df.tail(3)
"""
    ),
    md(
        """
## Walk-forward fold

Uses the frozen research config's expanding walk-forward windows (8y train /
2y val / 1y test, SPEC.md section 11) against the bounded real-data sample.
"""
    ),
    code(
        """
all_dates = long_df.index.get_level_values("date").unique()
folds = generate_walk_forward_folds(all_dates, research_config.walk_forward)
print(f"{len(folds)} walk-forward fold(s) available from this bounded sample")
for f in folds:
    print(f"  fold {f.index}: train {f.train_dates.min().date()}..{f.train_dates.max().date()} "
          f"({len(f.train_dates)}d) | val {f.val_dates.min().date()}..{f.val_dates.max().date()} "
          f"({len(f.val_dates)}d) | test {f.test_dates.min().date()}..{f.test_dates.max().date()} "
          f"({len(f.test_dates)}d)")

fold = folds[0]
train_df = select_dates(long_df, fold.train_dates)
val_df = select_dates(long_df, fold.val_dates)
test_df = select_dates(long_df, fold.test_dates)
print(f"\\nUsing fold 0 for this notebook: train={len(train_df)} val={len(val_df)} test={len(test_df)} rows")
"""
    ),
    md("## Train every required baseline and model on identical dates"),
    code(
        """
models = {
    "equal_weight": EqualWeightModel(),
    "rule_momentum": RuleMomentumModel(),
    "linear_ridge": RidgeBaselineModel(seed=research_config.seed),
    "tree": TreeModel(seed=research_config.seed),
    "mlp": MLPModel(seed=research_config.seed),
    "transparent_trend": TransparentTrendModel(),
}

test_scores: dict[str, pd.Series] = {}
for name, model in models.items():
    model.fit(train_df, val_df)
    test_scores[name] = model.predict(test_df)

test_forward_return = test_df["label_regression"]

ensemble_result = combine_ensemble(
    {k: test_scores[k] for k in ("tree", "mlp", "transparent_trend")}, research_config.ensemble
)
test_scores["ensemble"] = ensemble_result.score
print("Ensemble weights used:", ensemble_result.weights_used, "excluded:", ensemble_result.excluded)
"""
    ),
    md(
        """
## Model comparison (SPEC.md acceptance criteria #5)

Every model is scored on the **same out-of-sample test dates**. Rank IC and
top-quintile hit rate are the SPEC.md section 11 primary ML metrics scoped to
this five-market cross-section (see `frtbot.reporting.ml_metrics` docstring
for why Brier/regime-stability metrics are deferred to the M3 stock-level
slice, where the larger per-date cross-section makes them more informative).
"""
    ),
    code(
        """
comparison = pd.DataFrame(
    {
        "rank_ic": {name: rank_information_coefficient(s, test_forward_return) for name, s in test_scores.items()},
        "top_quintile_hit_rate": {name: top_quintile_hit_rate(s, test_forward_return) for name, s in test_scores.items()},
    }
).sort_values("rank_ic", ascending=False)
comparison
"""
    ),
]


# ---------------------------------------------------------------------------
# 02_walk_forward_backtest.ipynb
# ---------------------------------------------------------------------------

CELLS_02 = [
    md(
        """
# 02 - Walk-Forward Backtest

Runs the country allocation + monthly-rebalance backtest engine (SPEC.md
sections 9-10) over the walk-forward test fold from `01_global_country_model.ipynb`,
under all four cost scenarios, and reports gross/net portfolio metrics
(SPEC.md section 11) plus country attribution.

This is an **out-of-sample test-fold result on real bounded data** - not a
claim of achieved production performance. See README for the deferred
full-history multi-fold production command.
"""
    ),
    code(
        """
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from frtbot.config import load_markets_config, load_research_config
from frtbot.data.cache import DataCache
from frtbot.data.providers import get_provider
from frtbot.data.fx import align_fx_to_index, fetch_fx_series, identity_fx_series, to_thb_price
from frtbot.data.quality import detect_extreme_return_days
from frtbot.features.build import MarketSeries, build_country_feature_panel
from frtbot.labels.country import country_classification_label, country_regression_label
from frtbot.models.dataset import build_country_dataset, select_dates
from frtbot.backtest.splits import generate_walk_forward_folds
from frtbot.models.tree import TreeModel
from frtbot.models.mlp import MLPModel
from frtbot.models.trend import TransparentTrendModel
from frtbot.models.ensemble import combine_ensemble
from frtbot.portfolio.construction import construct_country_weights
from frtbot.backtest.engine import cash_daily_return_series, monthly_rebalance_dates, run_all_cost_scenarios
from frtbot.reporting.metrics import country_attribution, summarize_all_scenarios
from frtbot.seed import set_global_seed

pd.set_option("display.width", 140)
set_global_seed(42)

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "configs").exists():
    REPO_ROOT = REPO_ROOT.parent

START = date(2012, 6, 1)
END = date(2026, 7, 31)

markets_config = load_markets_config(REPO_ROOT / "configs" / "markets.example.yml")
research_config = load_research_config(REPO_ROOT / "configs" / "research.yml")
cache = DataCache(root=REPO_ROOT / "data")
provider = get_provider("yfinance")
"""
    ),
    code(
        """
market_series: dict[str, MarketSeries] = {}
for m in markets_config.markets:
    df, _ = cache.get_or_fetch(m.key, m.provider_symbol, "ohlcv", provider, START, END)

    # Data-quality gate (SPEC.md section 3): exclude markets with an implausible single-day
    # move - see notebook 00's audit, which flags this same check. Discovered in practice:
    # 1306.T (JP) has a two-day misadjustment (likely an unadjusted stock split) in the
    # upstream Yahoo Finance feed; including it corrupts volatility/return calculations.
    anomalies = detect_extreme_return_days(df["close"])
    if len(anomalies) > 0:
        print(f"EXCLUDING {m.key} ({m.provider_symbol}): price anomaly on {[d.date() for d in anomalies]}")
        continue

    if m.fx_pair is None:
        fx_aligned = identity_fx_series(df.index)
    else:
        fx_entry = markets_config.fx_for(m)
        rate, _ = fetch_fx_series(fx_entry, cache, provider, START, END)
        fx_aligned = align_fx_to_index(rate, df.index)
    market_series[m.key] = MarketSeries(key=m.key, ohlcv=df, fx_rate_aligned=fx_aligned)

print(f"Usable markets after data-quality gate: {sorted(market_series.keys())}")

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
fold = folds[0]
train_df = select_dates(long_df, fold.train_dates)
val_df = select_dates(long_df, fold.val_dates)
test_df = select_dates(long_df, fold.test_dates)
print(f"Test fold: {fold.test_dates.min().date()} .. {fold.test_dates.max().date()} ({len(fold.test_dates)} trading days)")
"""
    ),
    md("## Train the ensemble models on the fold and build monthly target weights"),
    code(
        """
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
tree_predicted_return = test_scores["tree"]  # eligibility gate uses the tree model's native return-scale prediction

rebalance_dates = monthly_rebalance_dates(fold.test_dates)
vol_63 = {k: panel[k]["vol_63d"] for k in panel}

target_weights = {}
for d in rebalance_dates:
    if d not in ensemble_result.score.index.get_level_values("date"):
        continue
    day_scores = ensemble_result.score.xs(d, level="date")
    day_pred = tree_predicted_return.xs(d, level="date")
    day_vol = pd.Series({k: vol_63[k].get(d, float("nan")) for k in vol_63})
    target_weights[d] = construct_country_weights(day_scores, day_pred, day_vol, research_config.portfolio)

print(f"{len(target_weights)} monthly rebalances in the test fold")
pd.DataFrame(target_weights).T.fillna(0.0).round(3)
"""
    ),
    md("## Backtest under all four cost scenarios (SPEC.md section 10)"),
    code(
        """
cash_ret = cash_daily_return_series(fold.test_dates, research_config.cash_annual_rate)
results = run_all_cost_scenarios(
    fold.test_dates, target_weights, thb_daily_return, cash_ret, markets_config, research_config.costs
)

summary = summarize_all_scenarios(results, risk_free_daily=cash_ret.iloc[0])
summary_cols = [
    "scenario", "cagr_net", "cumulative_return_net", "annualized_vol_net", "sharpe_net",
    "sortino_net", "max_drawdown_net", "total_turnover", "n_rebalances", "cost_drag",
]
summary[summary_cols]
"""
    ),
    code(
        """
fig, ax = plt.subplots(figsize=(9, 5))
for scenario, result in results.items():
    equity = (1 + result.daily_net_return).cumprod()
    ax.plot(equity.index, equity.values, label=f"{scenario} (net)")
ax.set_title("Out-of-sample test-fold equity curve by cost scenario (THB, net of costs)")
ax.set_ylabel("Growth of 1 THB")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
"""
    ),
    md("## Country attribution (base cost scenario)"),
    code(
        """
attribution = country_attribution(results["base"].daily_weights, thb_daily_return)
attribution.to_frame("gross_return_contribution")
"""
    ),
    md(
        """
## Limitations of this run

- Single walk-forward fold shown inline; see `folds` above for how many the
  bounded 2012-2026 real-data sample actually supports, and README for the
  exact command to run every fold end-to-end.
- Execution timing approximates "next tradable bar" as the next full
  close-to-close return rather than modeling intraday open-price execution
  (see `frtbot.backtest.engine` module docstring).
- `cash_annual_rate` is a documented flat-rate assumption, not a downloaded
  market series (see `configs/research.yml`).
- Five country proxies only - stock-level universes are M3, not this POC.
"""
    ),
]


def build_and_execute(name: str, cells: list, timeout: int = 1800) -> None:
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "name": "frtbot-venv",
        "display_name": "Python 3 (frtbot .venv)",
        "language": "python",
    }
    client = NotebookClient(nb, timeout=timeout, kernel_name="frtbot-venv", resources={"metadata": {"path": str(NOTEBOOKS_DIR)}})
    client.execute()
    out_path = NOTEBOOKS_DIR / name
    nbformat.write(nb, out_path)
    print(f"wrote executed notebook: {out_path}")


if __name__ == "__main__":
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    build_and_execute("00_data_audit.ipynb", CELLS_00)
    build_and_execute("01_global_country_model.ipynb", CELLS_01, timeout=3600)
    build_and_execute("02_walk_forward_backtest.ipynb", CELLS_02, timeout=3600)
