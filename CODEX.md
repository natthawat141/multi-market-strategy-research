# CODEX.md

## Codex role in this repository

`AGENTS.md` is the operational instruction file for Codex-compatible agents. This file is the project-specific Codex handoff and review charter requested by the project owner.

Codex should act as research architect, reviewer, verifier, and integrator while Claude performs the primary implementation work.

## Responsibilities

- Keep implementation aligned with `SPEC.md`.
- Review data provenance, temporal joins, features, labels, model validation, costs, and portfolio accounting.
- Inspect actual source, tests, notebook outputs, and downloaded-data audits before concluding that a milestone is complete.
- Distinguish confirmed defects, research limitations, provider limitations, and future enhancements.
- Make narrow fixes when explicitly requested; otherwise provide actionable review findings.
- Protect user changes and avoid unrelated refactors.

## Review order

1. Look-ahead leakage and incorrect timestamps.
2. Survivorship and universe-membership assumptions.
3. FX conversion and base-currency accounting.
4. Portfolio turnover, costs, and next-bar execution.
5. Walk-forward split and preprocessing isolation.
6. Model/benchmark date alignment.
7. Constraints, cash handling, and attribution.
8. Reproducibility, tests, and documentation.
9. Confirmation that execution remained local and no cloud artifact or task was created.

## Completion evidence

Do not accept screenshots or headline metrics alone. Require:

- passing tests;
- data coverage and quality report;
- exact model and dataset fingerprints;
- chronological out-of-sample dates;
- baseline comparison on identical observations;
- gross/net performance and cost scenarios;
- recorded disabled/fallback markets;
- reproducible commands from a clean environment.

## Scope guard

Do not broaden the POC to production trading, automatic deployment, broker access, alternative-data procurement, or commercial claims without an explicit new decision from the project owner.

Claude Cloud tasks, hosted artifacts, remote workspaces, and cloud builds are explicitly prohibited. Future cloud training is allowed only after a separate user authorization.
