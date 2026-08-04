# FRTBOT

Research proof of concept for a global multi-market ML/DL portfolio research system.
THB-denominated, monthly-rebalanced, long-only, research/paper-simulation only — **not
investment advice and not a live trading system**. See `SPEC.md` for the full product
and technical contract, and `AGENTS.md` / `CLAUDE.md` / `CODEX.md` for agent operating
rules.

Status: **M0 (foundation), M1 (five-market country vertical slice), and M2
(data-quality and reporting gates) are implemented and verified** against real
downloaded market data. M3 (stock-level adapters) and M4 (sequence model) are not
started — see "Milestones" below.

## Repository layout

```text
configs/           markets.example.yml (5 country proxies + FX), research.yml (windows, costs, ensemble weights)
src/frtbot/        the package: config, data, features, labels, models, portfolio, backtest, reporting
tests/             pytest suite; tests/fixtures/ holds the committed SYNTHETIC fixture dataset
notebooks/         00_data_audit, 01_global_country_model, 02_walk_forward_backtest (pre-executed, real data)
scripts/           build_notebooks.py (regenerates the notebooks), run_full_backtest.py (multi-fold report)
data/, artifacts/  gitignored - local cache and generated output only, never committed
```

## Windows PowerShell quickstart

```powershell
# 1. Create and activate a Python 3.12 virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install the package with the dependency groups you need
#    (data = yfinance + calendars, ml = scikit-learn/LightGBM/PyTorch, notebook = Jupyter, dev = pytest/ruff)
pip install -e ".[data,ml,notebook,dev]"

# 3. Run the test suite (fast, deterministic, uses only the committed SYNTHETIC fixtures - no network)
pytest -q

# 4. Lint
ruff check src tests

# 5. Regenerate the three report notebooks against real downloaded data
#    (first run registers a Jupyter kernel for this venv; only needed once per machine)
python -m ipykernel install --user --name frtbot-venv --display-name "Python 3 (frtbot .venv)"
python scripts/build_notebooks.py

# 6. Run the full multi-fold walk-forward backtest report to the console
python scripts/run_full_backtest.py

# 7. Export the detailed local evidence bundle and canonical report artifact
python scripts/export_detailed_backtest.py
python scripts/build_detailed_report_artifact.py
```

Steps 5-6 download real daily data via `yfinance` into the local, gitignored `data/`
cache on first run (subsequent runs reuse the cache; pass `force_refresh=True` to
`DataCache.get_or_fetch` to bypass it). Steps 1-4 need no network access beyond `pip
install`.

## Configuration

- `configs/markets.example.yml` — the five country proxies (SPEC.md section 2) and their
  FX pairs. Copy to `configs/markets.local.yml` (gitignored) for machine-specific
  overrides; the loader (`frtbot.config.load_markets_config`) validates every field and
  fails loudly on duplicate keys, currency/FX-pair mismatches, or a `disabled` market
  missing its reason.
- `configs/research.yml` — walk-forward windows, portfolio caps, the frozen ensemble
  weights (`0.40*tree + 0.30*mlp + 0.30*trend`), and the four cost scenarios.

## Data sources actually reached

All five country proxies and four FX pairs were **successfully downloaded from Yahoo
Finance** (`yfinance`, public/open data) for 2012-06-01 through 2026-07-30:

| Market | Provider symbol | Currency | FX pair (provider symbol) |
|---|---|---|---|
| US | `SPY` | USD | `USDTHB=X` |
| EU | `EXSA.DE` | EUR | `EURTHB=X` |
| JP | `1306.T` | JPY | `JPYTHB=X` |
| CN | `510300.SS` | CNY | `THBCNY=X`, inverted (no direct `CNYTHB=X` quote) |
| TH | `TDEX.BK` | THB | n/a (already base currency) |

Run `python scripts/build_notebooks.py` and open `notebooks/00_data_audit.ipynb` for the
full coverage/staleness/adjustment report.

### Discovered data-quality defect: JP (`1306.T`)

The data-quality gate (`frtbot.data.quality.detect_extreme_return_days`, wired into the
data audit and every modeling script/notebook) flags implausible single-day price moves
for country-index proxies (threshold: 40% in one day — broad indices essentially never
move that much in one session; see `frtbot/data/quality.py` for the reasoning). It found
that Yahoo Finance's `1306.T` (Nomura TOPIX ETF) feed has a **reproducible** two-day
misadjustment around 2026-03-30/2026-03-31 (price drops to ~1/10th, then jumps back the
next session — almost certainly an unadjusted stock split), plus another anomaly on
2015-01-05. This was confirmed on a fresh re-fetch (not a transient lag). **JP is
therefore excluded from every modeling/backtest run in this repository** with a printed,
visible reason — this is the SPEC.md section 3 `disabled`-with-reason principle applied
dynamically based on the audit, rather than assuming any downloaded series is trustworthy
by default. A production system would need a corporate-action-aware data provider before
trusting `1306.T` (or any single free-tier feed) unattended.

## Known limitations / documented design choices

- **Execution timing**: a signal computed from data through close of day `t` selects
  weights that take effect at the next tradable date `t+1`'s full close-to-close return
  (never `t`'s own return — verified in `tests/test_no_lookahead.py`). True intraday
  open-price execution (SPEC.md section 10) is not modeled. See
  `frtbot.backtest.engine` module docstring.
- **Cash proxy**: `cash_annual_rate` (`configs/research.yml`) is a documented flat-rate
  assumption, not a downloaded market series (no reliable free THB short-rate feed was
  used) — always reported as such, never presented as real market data.
- **Cross-market eligibility gate**: "positive predicted THB excess return" (SPEC.md
  9.1) uses the **tree model's** native regression output; the frozen ensemble rank
  score (SPEC.md 8.4) ranks/selects among already-eligible markets. SPEC.md leaves this
  ambiguous; see `frtbot.backtest.engine` module docstring.
- **Cross-market same-date comparison**: five different exchanges (NYSE, Xetra, TSE,
  SSE, SET) close on different local holidays. Same-date cross-sectional features
  (relative momentum rank, global risk regime, rolling correlation to the global
  return) are computed as of each market's own trading-date index rather than a shared
  intraday-synchronized calendar — a standard daily-bar-level approximation, called out
  explicitly since a naive implementation of the correlation feature was found (and
  fixed, see `frtbot.features.market_context.rolling_corr_with_global`) to silently
  degrade almost entirely to `NaN` under exactly this kind of calendar mismatch.
- **ML metrics scope**: SPEC.md section 11 lists rank IC, top-quintile precision/recall,
  Brier score, and regime stability. At the five-market country-level cross-section,
  rank IC and top-quintile hit rate are implemented and meaningful
  (`frtbot.reporting.ml_metrics`); Brier/calibration and regime-stability are deferred to
  the M3 stock-level slice, where a much larger per-date cross-section makes them
  informative rather than degenerate.
- **Stock-mode, fundamentals, news, sequence models, live trading**: all explicitly out
  of scope for this POC (SPEC.md milestones M3/M4 and non-goals).

## Reproducible reports

- `notebooks/00_data_audit.ipynb` — downloads/caches all 5 markets + 4 FX pairs, runs
  the data audit (coverage, gaps, staleness, adjustment metadata, price-anomaly flags).
- `notebooks/01_global_country_model.ipynb` — builds the leakage-safe feature/label
  panel, runs one walk-forward fold, trains every required baseline plus
  tree/MLP/transparent-trend, and compares them on identical out-of-sample dates via
  rank IC and top-quintile hit rate.
- `notebooks/02_walk_forward_backtest.ipynb` — country allocation + monthly-rebalance
  backtest for that fold under all four cost scenarios, with country attribution.
- `scripts/run_full_backtest.py` — the most complete result: trains and backtests
  **every** walk-forward fold the real 2012-2026 sample supports (5 folds, using the
  frozen 8y-train/2y-val/1y-test/1y-step config from `configs/research.yml` — this is
  the actual production window, not a reduced smoke config) and stitches the
  consecutive out-of-sample test segments into one continuous multi-year result.
- `scripts/export_detailed_backtest.py` — re-runs all folds and writes the complete
  local evidence bundle under `artifacts/detailed_report/data/`: daily strategy and
  benchmark returns, weights, fold/model metrics, audit rows, feature inventory, and
  the full modeling panel. It also adds equal-weight, cash, and per-market buy-and-hold
  comparisons without changing the frozen FRTBOT strategy.
- `scripts/build_detailed_report_artifact.py` — converts those reviewed exports into
  the canonical Data Analytics `artifact.json` used to build the self-contained Thai
  technical report. Generated report/data files remain gitignored and local-only.

Last run of `scripts/run_full_backtest.py` (4 usable markets after the JP data-quality
exclusion above; continuous out-of-sample period 2022-06-01 to 2026-07-30, 1084 trading
days, net of costs):

| Scenario | CAGR | Cumulative return | Ann. vol | Sharpe | Sortino | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Zero | 9.23% | 46.2% | 11.08% | 0.74 | 1.26 | -15.7% |
| Optimistic | 8.92% | 44.4% | 11.08% | 0.71 | 1.21 | -15.8% |
| Base | 8.61% | 42.6% | 11.08% | 0.69 | 1.17 | -15.9% |
| Severe | 7.67% | 37.4% | 11.09% | 0.61 | 1.03 | -16.3% |

This is a real out-of-sample walk-forward result, not a backtest overfit to the full
history (each fold's models only ever see data available before that fold's own test
period) — and not a claim of achieved or future performance (SPEC.md non-goals). Fold 0
alone (2022-06 to 2023-05, shown in detail in notebook 02) was a losing period; the
full multi-year result is positive. Both are preserved and reported, per SPEC.md
section 11 ("preserve every final out-of-sample result, including failures").

## Milestones

- **M0 — Project foundation**: done. `pyproject.toml`, `src/` layout, Ruff, pytest,
  pydantic-validated config schema, deterministic seeding, committed SYNTHETIC fixtures.
- **M1 — Five-market country vertical slice**: done. Data provider + immutable-raw/
  normalized cache, FX-to-THB conversion, full SPEC.md section 6 feature set, section 7
  labels, deterministic expanding walk-forward splits, all required baselines plus
  LightGBM tree / small MLP / transparent trend, frozen ensemble, country portfolio
  construction with caps, monthly-rebalance backtest with 4 cost scenarios.
- **M2 — Data-quality and reporting gates**: done. Data audit (usable/proxy/disabled/
  missing/stale/synthetic classification + price-anomaly plausibility gate), leakage/
  FX/constraint/cost/split test coverage, portfolio metrics (CAGR, vol, Sharpe, Sortino,
  drawdown+recovery, turnover, cost drag, worst month/year, country attribution).
- **M3 — Stock-mode adapters**: not started (explicitly deferred; proxy mode is a
  first-class supported state per SPEC.md).
- **M4 — Sequence model research**: not started.

## Confirmation

All work was performed locally under `D:\code\FRTBOT`. No Claude Cloud task, remote
workspace, hosted artifact, deployment, or upload to any cloud service was created at
any point. Network access was used only for `pip install` and the documented public
Yahoo Finance downloads.
