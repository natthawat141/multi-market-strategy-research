# CLAUDE.md

## Your assignment

You are the primary implementation agent for FRTBOT. Read `SPEC.md` and `AGENTS.md` completely before writing code.

Implement the frozen POC in milestone order. Begin with M0, M1, and M2: project foundation, the five-market country-level vertical slice, and the validation/reporting gates. Do not jump to live trading, news, fundamentals, stock universes, or a complex UI before the vertical slice is verified.

All work is local-only in `D:\code\FRTBOT`. Do not create Claude Cloud tasks, remote workspaces, hosted artifacts, deployments, or cloud builds. Do not upload project files or data. Public-data and package downloads may be used only to support local execution.

If full model training or a complete backtest would be too resource-intensive for this PC, implement the pipeline, run unit tests and a bounded smoke sample, then document the exact deferred full-run command. Do not force the heavy run.

## Required first-turn workflow

1. Inspect the repository and current working tree.
2. Summarize the M0-M2 implementation plan and list any discovered conflicts.
3. Scaffold the Python package and tests.
4. Implement one end-to-end thin slice before broadening abstractions.
5. Run tests and a small data smoke test.
6. Continue until M0-M2 acceptance criteria are satisfied or a concrete external blocker is proven.

Do not stop merely to ask which library to choose when `SPEC.md` permits a reasonable implementation decision. Record material choices in an architecture note or README.

## Implementation preferences

- Python 3.12, `pyproject.toml`, `src/` layout, pytest, Ruff, and type hints.
- Pandas/NumPy or Polars for canonical data transforms; use one consistently in core logic.
- PyArrow/Parquet for storage.
- scikit-learn-compatible model interfaces.
- LightGBM when installation is reliable; XGBoost is an acceptable documented fallback.
- PyTorch for MLP and later TCN/1D-CNN.
- Exchange-calendar-aware dates and timezone-safe timestamps.
- Plotly or Matplotlib for notebook reports.

Keep dependencies smaller than the research question requires. Do not add a framework simply because it is popular.

## Definition of done for your initial implementation

- All five market proxies and FX mappings are configuration-driven.
- A data audit distinguishes usable, proxy, disabled, missing, stale, and synthetic series.
- Features and labels follow `SPEC.md` without same-bar leakage.
- Expanding walk-forward splits are deterministic.
- Baseline, tree, MLP, and transparent trend models share identical test periods.
- Ensemble predictions are rank-normalized and use the frozen weights.
- Portfolio construction respects cash and country caps.
- Costs are applied to turnover under all four scenarios.
- Tests exercise temporal, FX, constraint, and cost invariants.
- At least one notebook or CLI run produces a reproducible report.
- README contains exact Windows PowerShell setup and run commands.

## Things you must not do

- Do not infer or imitate proprietary FRTST parameters.
- Do not use random splits or fit preprocessing on the full dataset.
- Do not silently replace missing market data with random/mock data.
- Do not optimize ensemble weights on the final test set.
- Do not call an in-sample chart an out-of-sample result.
- Do not implement broker connectivity or real-money execution.
- Do not commit downloaded data, secrets, virtual environments, or generated artifacts.
- Do not invoke Claude Cloud, remote execution, hosted artifacts, deployments, or any cloud provider.
- Do not run resource-heavy training merely to claim completion; defer it transparently when local capacity is insufficient.

## Handoff format

At each meaningful checkpoint report:

1. files changed;
2. behavior implemented;
3. exact commands run and results;
4. data sources actually reached;
5. remaining limitations or failed checks;
6. next milestone.
7. confirmation that no cloud task, upload, deployment, or hosted artifact was created.
