from __future__ import annotations

import pandas as pd
import pytest

from frtbot.reporting.ml_metrics import rank_information_coefficient, top_quintile_hit_rate


def _perfect_rank_panel() -> tuple[pd.Series, pd.Series]:
    idx = pd.MultiIndex.from_tuples(
        [("d1", "A"), ("d1", "B"), ("d1", "C"), ("d2", "A"), ("d2", "B"), ("d2", "C")],
        names=["date", "market"],
    )
    score = pd.Series([1, 2, 3, 30, 20, 10], index=idx)
    forward_return = pd.Series([0.01, 0.02, 0.03, 0.03, 0.02, 0.01], index=idx)
    return score, forward_return


def test_rank_ic_is_perfect_for_monotonic_relationship():
    score, forward_return = _perfect_rank_panel()
    ic = rank_information_coefficient(score, forward_return)
    assert ic == pytest.approx(1.0)


def test_rank_ic_is_negative_for_inverted_relationship():
    score, forward_return = _perfect_rank_panel()
    inverted_score = -score
    ic = rank_information_coefficient(inverted_score, forward_return)
    assert ic == pytest.approx(-1.0)


def test_top_quintile_hit_rate_perfect_when_top_pick_matches():
    score, forward_return = _perfect_rank_panel()
    hit_rate = top_quintile_hit_rate(score, forward_return, quintile=0.2)
    assert hit_rate == pytest.approx(1.0)


def test_top_quintile_hit_rate_zero_when_top_pick_never_matches():
    idx = pd.MultiIndex.from_tuples(
        [("d1", "A"), ("d1", "B"), ("d1", "C")], names=["date", "market"]
    )
    score = pd.Series([3, 2, 1], index=idx)  # picks A
    forward_return = pd.Series([0.01, 0.02, 0.03], index=idx)  # best is C
    hit_rate = top_quintile_hit_rate(score, forward_return, quintile=0.2)
    assert hit_rate == pytest.approx(0.0)
