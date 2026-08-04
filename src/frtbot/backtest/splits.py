"""Deterministic expanding walk-forward splits (SPEC.md section 11).

- Never a random split: folds are pure date arithmetic over a sorted trading
  calendar, so the same inputs always produce the same folds.
- "Expanding": the training window start is fixed at the start of history and
  its end grows by `step_years` each fold; validation and test are
  fixed-length windows that slide forward with it.
- Embargo: the last `embargo_days` trading days of the train window (before
  validation) and of the validation window (before test) are dropped, so no
  training/validation label's forward-return horizon overlaps the next
  segment (SPEC.md section 7.1: "purged or embargoed at validation
  boundaries").
- If history is insufficient for the configured window lengths, windows are
  shrunk transparently down to `min_train_years`/1-year val/1-year test
  rather than silently mixing future data into training.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from frtbot.config import WalkForwardConfig


@dataclass(frozen=True)
class Fold:
    index: int
    train_dates: pd.DatetimeIndex
    val_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex


def _slice_between(dates: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return dates[(dates >= start) & (dates < end)]


def _drop_tail_days(dates: pd.DatetimeIndex, n: int) -> pd.DatetimeIndex:
    if n <= 0 or len(dates) == 0:
        return dates
    return dates[: max(0, len(dates) - n)]


def generate_walk_forward_folds(
    dates: pd.DatetimeIndex, config: WalkForwardConfig
) -> list[Fold]:
    dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    if len(dates) == 0:
        return []

    history_start = dates.min()
    history_end = dates.max()
    span_years = (history_end - history_start).days / 365.25

    train_years, val_years, test_years = (
        config.train_years,
        config.val_years,
        config.test_years,
    )
    if span_years < (train_years + val_years + test_years):
        train_years = max(config.min_train_years, 1)
        val_years = 1
        test_years = 1
        if span_years < (train_years + val_years + test_years):
            return []

    folds: list[Fold] = []
    fold_index = 0
    while True:
        train_end = history_start + pd.DateOffset(
            years=train_years + fold_index * config.step_years
        )
        val_start = train_end
        val_end = val_start + pd.DateOffset(years=val_years)
        test_start = val_end
        test_end = test_start + pd.DateOffset(years=test_years)

        if test_start > history_end:
            break

        train_dates = _drop_tail_days(
            _slice_between(dates, history_start, train_end), config.embargo_days
        )
        val_dates = _drop_tail_days(
            _slice_between(dates, val_start, val_end), config.embargo_days
        )
        test_dates = _slice_between(dates, test_start, min(test_end, history_end + pd.Timedelta(days=1)))

        if len(train_dates) == 0 or len(val_dates) == 0 or len(test_dates) == 0:
            fold_index += 1
            if test_end > history_end:
                break
            continue

        folds.append(Fold(fold_index, train_dates, val_dates, test_dates))

        if test_end >= history_end:
            break
        fold_index += 1

    return folds
