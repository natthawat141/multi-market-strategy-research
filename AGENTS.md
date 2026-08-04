# AGENTS.md

This file applies to the entire repository.

## Canonical direction

Read `SPEC.md` before planning or editing. `SPEC.md` is the canonical product and technical contract. If another document conflicts with it, stop and surface the conflict instead of inventing a compromise.

FRTBOT is a research-only global ML/DL quantitative backtesting system. It is not a live trading bot and does not reproduce FRTST.

## Safety boundaries

- Do not add broker credentials, broker APIs, live orders, leverage, shorting, or production trading automation.
- Work locally in this repository only. Do not create cloud tasks, remote workspaces, hosted artifacts, deployments, or provider-managed builds.
- Do not upload source, data, notebooks, model files, reports, or artifacts to Claude Cloud or any other cloud service.
- Network access is limited to documented open-data downloads and local dependency installation.
- If training or a backtest is too heavy for the local PC, use a bounded fixture or small real-data smoke sample, leave the full-run command documented, and do not execute the heavy job.
- Never commit secrets, downloaded licensed datasets, large model artifacts, or user-specific absolute paths.
- Mock data is allowed only in tests and demonstrations. Mark every mock output `SYNTHETIC`.
- Do not present a backtest as investment advice or guaranteed performance.

## Quantitative correctness rules

- No random train/test splits for time-series research.
- No feature may use information published after its observation time.
- A feature using a market close may trade no earlier than the next valid local-market bar.
- Forward labels must never enter features or preprocessing statistics.
- Fit scalers, imputers, feature selectors, and models on training data only.
- Purge or embargo overlapping forward-label windows at validation boundaries.
- Use point-in-time universe membership where available and disclose approximations.
- Preserve inactive/delisted securities where the source permits.
- Measure returns in THB for primary portfolio results and retain local-currency diagnostics.
- Report costs, turnover, missing data, and disabled markets explicitly.

## Engineering rules

- Target Python 3.12 unless an actual dependency requires a documented downgrade.
- Keep reusable code in `src/frtbot`; notebooks should call package functions.
- Prefer typed, small, testable functions and deterministic random seeds.
- Use configuration files for symbols, providers, calendars, thresholds, model settings, and costs.
- Keep raw downloads immutable. Store normalized and derived tabular data as Parquet.
- Use stable internal security identifiers rather than ticker alone.
- Cache downloads but provide a force-refresh option.
- Fail loudly on ambiguous currency, timezone, duplicate keys, non-monotonic timestamps, or invalid adjustment metadata.
- Preserve existing user changes and avoid unrelated rewrites.

## Required verification

Before claiming a change complete, run the narrowest relevant checks and then the full available suite. At minimum, changes affecting research logic require tests for:

- temporal leakage and signal shifting;
- FX return conversion;
- portfolio weight constraints;
- transaction-cost application;
- deterministic walk-forward splits;
- data schema and duplicate-key validation.

Record commands and outcomes in the handoff. If a dependency or data provider prevents validation, state the exact limitation and do not mark the feature complete.

Also verify that no cloud task, deployment, upload, or hosted artifact was created.

## Documentation

- Update `README.md` for user-facing commands.
- Update `SPEC.md` only when a product decision genuinely changes; append the decision to its decision log.
- Document data source, license/usage caveat, retrieval date, field meaning, and known bias for each adapter.
- Keep `CLAUDE.md` and `CODEX.md` role guidance aligned with this file.

## First implementation priority

Implement milestones M0, M1, and M2 from `SPEC.md` before stock-level expansion, fundamentals, news, sequence models, dashboards, or deployment.
