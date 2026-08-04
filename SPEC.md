# FRTBOT Global Multi-Market ML/DL Research POC

Status: Ready for implementation  
Specification version: 0.1.0  
Base currency: THB  
Primary rebalance frequency: Monthly  
Purpose: Research and paper simulation only

Execution boundary: Local workspace only

## 1. Product intent

FRTBOT is an original quantitative research system. It is not an attempt to copy Fortune Robotics Trading Super Trend or recover any proprietary formula.

The system will combine transparent trend features with machine-learning and deep-learning models to:

1. rank five equity markets;
2. rank stocks inside eligible markets when reliable stock-level data is available;
3. allocate a long-only portfolio under explicit risk limits;
4. backtest every decision using point-in-time-safe, next-tradable-bar assumptions; and
5. report both successful and unsuccessful out-of-sample results.

The first release is a research proof of concept, not investment advice, an order-management system, or a live trading robot.

## 2. Market scope

The global layer must support all five markets from the first vertical slice:

| Market | Country layer | Preferred stock universe | Currency |
|---|---|---|---|
| United States | Broad US or S&P 500 proxy | Point-in-time S&P 500 | USD |
| Europe | Broad Europe or STOXX Europe proxy | Point-in-time STOXX Europe 600 | EUR/mixed |
| Japan | TOPIX or broad Japan proxy | Point-in-time TOPIX 500 | JPY |
| China | CSI 300 A-share proxy | Point-in-time CSI 300 | CNY |
| Thailand | SET100/SET50 TRI proxy | Archived SET100 membership | THB |

China means mainland A-shares for v0.1. Hong Kong and offshore Chinese listings are separate future universes and must not be silently mixed into CSI 300.

Every instrument mapping must live in configuration, not in model code. Each configured series must record provider, provider symbol, market, currency, timezone, exchange calendar, price type, adjustment type, first valid date, and last successful refresh.

## 3. Delivery modes

Each market has an explicit data mode:

- `stock`: stock-level point-in-time universe is available and passes validation;
- `proxy`: the country index or ETF proxy is used because stock-level history is incomplete;
- `disabled`: data quality checks failed and the market is excluded with a visible reason.

The POC must never substitute mock data for a missing live or historical market series in a reported backtest. Mock data is allowed only in unit tests and UI/pipeline demonstrations, and all mock outputs must be labelled `SYNTHETIC`.

## 3.1 Local-only development boundary

- All development, commands, notebooks, tests, caches, models, and generated reports must stay on the local machine under the repository or an explicitly configured local cache.
- Claude must not create or run Claude Cloud tasks, remote workspaces, hosted artifacts, deployments, or provider-managed builds.
- Do not upload source code, datasets, notebooks, model weights, reports, credentials, or artifacts to any cloud service.
- Network access is permitted only for downloading documented public/open research data and Python packages required by the local environment.
- Cloud training may be designed as a future deployment target, but v0.1 must not provision, authenticate to, or execute on a cloud platform.
- If a full training or backtest run would materially strain the local PC, implement and test the code with a small deterministic fixture or bounded real-data sample, record the exact deferred command, and leave full training unexecuted for a later user-authorized cloud run.

## 4. Frozen v0.1 research question

At each month-end, using only information known by that time:

1. Which markets are most likely to produce positive THB-denominated excess returns over the next 21 trading days?
2. Inside each eligible stock-mode market, which stocks are most likely to rank in the top return quintile relative to their local market over the next 21 trading days?
3. Does the resulting portfolio improve net risk-adjusted results or maximum drawdown versus simple benchmarks after realistic costs?

## 5. Data requirements

### 5.1 Required for the first vertical slice

- Daily raw OHLCV where available.
- Daily adjusted close or total-return proxy.
- Local exchange trading calendar and timezone.
- Daily FX rate from local currency to THB.
- Historical country proxy prices.
- Provider metadata and download timestamps.

### 5.2 Required before a market may use stock mode

- Historical universe membership or a clearly disclosed approximation.
- Stable internal security identifier independent of ticker.
- Delisted and inactive securities where obtainable.
- Corporate-action handling.
- Liquidity fields.
- Sufficient history for all required features.

Fundamentals, earnings events, and news are deferred from the first vertical slice. Their future adapters must preserve the public release timestamp, not only the reporting period.

### 5.3 Storage rules

- Raw downloads are immutable and partitioned by provider and retrieval date.
- Normalized data is stored as Parquet.
- Derived features and labels are versioned by a deterministic dataset fingerprint.
- Raw, cache, model, and report artifacts are not committed to Git.
- A small deterministic fixture dataset is committed only for tests.

## 6. Feature specification

All close-derived features observed on date `t` become tradable no earlier than the next valid local-market bar.

### 6.1 Price and trend features

- Returns over 5, 21, 63, 126, and 252 trading days.
- Conventional 12-to-1-month momentum.
- Distance from SMA 20, 50, 100, and 200.
- SMA 50 and SMA 200 slopes.
- EMA 50 minus EMA 200, normalized by price.
- 20-day and 55-day Donchian position/breakout.
- RSI 14 and ADX 14 where inputs are valid.
- Distance from 52-week high.

### 6.2 Risk and liquidity features

- Realized volatility over 20, 63, and 126 days.
- Downside volatility.
- Maximum drawdown over 63 and 252 days.
- Average traded value and volume z-score.
- Zero-volume/stale-price rate.

### 6.3 Market-context features

- Local market trend and momentum.
- Fraction of the local universe above SMA 200 when stock data permits.
- Cross-market relative momentum rank.
- Local currency return versus THB.
- Rolling market correlations and global risk regime.

### 6.4 Normalization

Stock features are winsorized using trailing or same-date cross-sectional information only, then converted to percentile ranks within country and date. Country identity is retained as a categorical feature or embedding. Raw absolute values may be retained alongside ranks when economically meaningful.

## 7. Labels

### 7.1 Country model

Primary regression label:

```text
next_21d_country_return_thb - next_21d_cash_return_thb
```

Secondary classification label:

```text
1 when next 21-day THB excess return is positive, otherwise 0
```

### 7.2 Stock model

Primary ranking/classification label:

```text
1 when a stock is in the top 20% of next 21-day local-market excess returns,
otherwise 0
```

Labels with overlapping 21-day horizons must be purged or embargoed at validation boundaries.

## 8. Models and ensemble

Every advanced model must be compared against simple baselines.

### 8.1 Required baselines

- Equal-weight country portfolio.
- Buy-and-hold each country proxy.
- Rule-based absolute plus relative momentum.
- Logistic/linear model with regularization.

### 8.2 Required ML/DL models

- LightGBM or XGBoost gradient-boosted trees as the primary tabular model.
- Small MLP neural network with regularization, early stopping, and deterministic seeds.
- Transparent rule-based trend score.

### 8.3 Experimental model

- TCN or 1D-CNN sequence model.

The sequence model runs in shadow research mode and receives no portfolio weight until it beats the required baselines out of sample and passes stability gates. Reinforcement learning and LLM price prediction are out of scope for v0.1.

### 8.4 Frozen ensemble formula

Convert each model output to a same-date percentile rank before combining:

```text
ensemble_score =
    0.40 * tree_model_rank
  + 0.30 * mlp_model_rank
  + 0.30 * transparent_trend_rank
```

These are fixed research starting weights, not optimized weights. A model that fails validation is excluded and the remaining weights are renormalized, with the exclusion reported.

## 9. Portfolio construction

### 9.1 Country allocation

- Consider only markets with positive predicted THB excess return and valid data.
- Select up to the top three eligible markets.
- Start from inverse 63-day volatility weights.
- Cap a country at 40%.
- Unallocated capital remains in the configured cash proxy.

### 9.2 Stock allocation

- In stock mode, select approximately 3-5 stocks per selected country.
- Target 15-25 stocks globally when enough valid candidates exist.
- Select only stocks above the configured liquidity threshold.
- Size by inverse volatility within the country allocation.
- Cap a stock at 10%, a country at 40%, and a sector at 25%.
- If insufficient candidates pass, hold the remainder in the country proxy or cash according to configuration; report which fallback was used.

The number of holdings is an output of eligibility and constraints, not a promise to always hold a fixed count.

## 10. Backtest protocol

- Base currency is THB; local-currency results are diagnostic only.
- Rebalance monthly after all relevant local markets have closed.
- Execute at the next tradable local-market open when open prices exist; otherwise use a documented next-bar approximation.
- Never trade at the same close used to calculate a signal.
- Charge costs on absolute position turnover.
- Run zero, optimistic, base, and severe cost scenarios.
- Include commission, spread/slippage, and configurable market-impact assumptions.
- No shorting, leverage, derivatives, broker API, or live orders in v0.1.

Default research cost assumptions per side:

| Scenario | Country proxy | Developed stock | Emerging stock |
|---|---:|---:|---:|
| Zero | 0 bps | 0 bps | 0 bps |
| Optimistic | 5 bps | 10 bps | 20 bps |
| Base | 10 bps | 25 bps | 50 bps |
| Severe | 25 bps | 50 bps | 100 bps |

## 11. Validation

- Never use a random train/test split.
- Use expanding walk-forward validation.
- Initial target windows: 8 years train, 2 years validation, 1 year test, advancing one year where history permits.
- If a market has insufficient history, reduce windows transparently rather than silently mixing future data.
- Freeze model hyperparameters before the final out-of-sample segment.
- Preserve every final out-of-sample result, including failures.
- Test neighboring feature horizons and portfolio thresholds; prefer broad stable plateaus over the best single cell.

Primary ML metrics:

- rank information coefficient;
- top-quintile precision and recall;
- calibration/Brier score for classifiers;
- prediction stability by country and regime.

Primary portfolio metrics:

- CAGR and cumulative return;
- annualized volatility;
- Sharpe and Sortino ratios;
- maximum drawdown and recovery duration;
- turnover and cost drag;
- worst month/year;
- country, sector, and model attribution;
- gross and net results.

## 12. Repository architecture

Claude should scaffold the following shape unless an implementation constraint is documented:

```text
FRTBOT/
├── AGENTS.md
├── CLAUDE.md
├── CODEX.md
├── SPEC.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── markets.example.yml
│   └── research.yml
├── notebooks/
│   ├── 00_data_audit.ipynb
│   ├── 01_global_country_model.ipynb
│   └── 02_walk_forward_backtest.ipynb
├── src/frtbot/
│   ├── config.py
│   ├── data/
│   ├── features/
│   ├── labels/
│   ├── models/
│   ├── portfolio/
│   ├── backtest/
│   └── reporting/
├── tests/
│   ├── fixtures/
│   ├── test_features.py
│   ├── test_no_lookahead.py
│   ├── test_fx_returns.py
│   ├── test_portfolio_constraints.py
│   └── test_backtest_costs.py
├── data/                 # ignored
└── artifacts/            # ignored
```

Notebooks are orchestration and explanation layers. Reusable logic belongs in `src/frtbot`, with tests.

## 13. Implementation milestones

### M0 - Project foundation

- Python project, dependency groups, linting, tests, configuration loading, logging, and deterministic seeds.
- Market configuration schema and synthetic fixtures.

### M1 - Five-market country vertical slice

- Download and validate five country proxies plus FX.
- Build THB returns and features.
- Train baselines, tree model, and MLP under walk-forward splits.
- Produce country allocations and a monthly net backtest.

### M2 - Data-quality and reporting gates

- Data audit notebook.
- Missing/stale data report.
- Look-ahead tests, cost scenarios, portfolio metrics, and benchmark report.

### M3 - Stock-mode adapters

- Add point-in-time universe adapters market by market.
- Start with the market having the cleanest verified data.
- Never delay M1 waiting for all five stock universes; proxy mode is a first-class supported state.

### M4 - Sequence model research

- Add TCN/1D-CNN dataset and model.
- Run in shadow mode and evaluate against the frozen M1 models.

## 14. MVP acceptance criteria

The POC is complete only when:

1. A clean environment can install and run the documented commands on Windows/PowerShell.
2. All five markets are represented as `stock`, `proxy`, or visibly `disabled` with a reason.
3. The data audit reports coverage, gaps, staleness, currency, and adjustment metadata.
4. Walk-forward splits are chronological and tested against leakage.
5. Tree, MLP, transparent trend, and simple benchmarks are compared on identical dates.
6. The portfolio obeys all stock, sector, and country limits.
7. Gross and net results are shown under four cost scenarios.
8. At least one notebook runs from start to finish using downloaded open data or committed fixtures.
9. Tests cover FX conversion, signal shifting, costs, constraints, and look-ahead prevention.
10. Reports clearly label proxy, incomplete, synthetic, and out-of-sample data.

## 15. Non-goals

- Reproducing FRTST returns or proprietary logic.
- Claiming guaranteed profitability.
- Live broker integration or automated order placement.
- High-frequency or intraday trading.
- Optimizing thousands of indicator combinations.
- Treating mock data as investment evidence.
- Hiding failed models or unfavorable periods.
- Creating Claude Cloud tasks, hosted artifacts, remote builds, or cloud deployments.
- Running resource-heavy full training when the local machine cannot support it safely.

## 16. Decision log

- Five markets are mandatory at the country layer.
- China v0.1 is mainland A-shares/CSI 300, not a blended China universe.
- THB is the investor base currency.
- Monthly 21-trading-day prediction is the primary horizon.
- Tree + MLP + transparent trend is the production research ensemble.
- TCN/1D-CNN is experimental until it earns inclusion out of sample.
- Open data is acceptable for POC research; every limitation must be surfaced.
- Stock-level coverage can be added incrementally without blocking the five-market country POC.
- Development is local-only; future cloud training requires a separate explicit decision.
