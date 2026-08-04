from __future__ import annotations

import pandas as pd
import pytest

from frtbot.config import PortfolioConfig
from frtbot.portfolio.construction import apply_country_cap, construct_country_weights


def test_apply_country_cap_redistributes_excess_when_possible():
    weights = pd.Series({"A": 0.6, "B": 0.25, "C": 0.15})
    capped = apply_country_cap(weights, cap=0.40)
    assert capped["A"] == pytest.approx(0.40)
    assert (capped <= 0.40 + 1e-9).all()
    assert capped.sum() == pytest.approx(1.0)  # fully redistributable with 2 free names


def test_apply_country_cap_leaves_residual_when_not_redistributable():
    weights = pd.Series({"A": 0.7, "B": 0.3})
    capped = apply_country_cap(weights, cap=0.40)
    assert capped["A"] == pytest.approx(0.40)
    assert capped["B"] == pytest.approx(0.40)
    assert capped.sum() == pytest.approx(0.80)  # 0.20 must go unallocated (-> cash)


def test_construct_country_weights_selects_top_n_by_score():
    scores = pd.Series({"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6, "E": 0.5})
    predicted = pd.Series({"A": 0.02, "B": 0.02, "C": 0.02, "D": 0.02, "E": 0.02})
    vol = pd.Series({"A": 0.1, "B": 0.1, "C": 0.1, "D": 0.1, "E": 0.1})
    cfg = PortfolioConfig(top_n_countries=3, max_country_weight=0.40)

    weights = construct_country_weights(scores, predicted, vol, cfg)
    selected = [k for k in weights.index if k != cfg.cash_key]
    assert set(selected) == {"A", "B", "C"}
    assert len(selected) == 3


def test_construct_country_weights_excludes_negative_predicted_return():
    scores = pd.Series({"A": 0.9, "B": 0.8})
    predicted = pd.Series({"A": -0.01, "B": 0.02})  # A fails the positive-return eligibility gate
    vol = pd.Series({"A": 0.1, "B": 0.1})
    cfg = PortfolioConfig(top_n_countries=3, max_country_weight=0.40)

    weights = construct_country_weights(scores, predicted, vol, cfg)
    assert "A" not in weights.index
    assert weights["B"] == pytest.approx(0.40)  # capped, remainder to cash
    assert weights[cfg.cash_key] == pytest.approx(0.60)


def test_construct_country_weights_all_ineligible_is_all_cash():
    scores = pd.Series({"A": 0.9, "B": 0.8})
    predicted = pd.Series({"A": -0.01, "B": -0.02})
    vol = pd.Series({"A": 0.1, "B": 0.1})
    cfg = PortfolioConfig()

    weights = construct_country_weights(scores, predicted, vol, cfg)
    assert list(weights.index) == [cfg.cash_key]
    assert weights[cfg.cash_key] == pytest.approx(1.0)


def test_construct_country_weights_inverse_vol_sizing():
    scores = pd.Series({"A": 0.9, "B": 0.8})
    predicted = pd.Series({"A": 0.02, "B": 0.02})
    vol = pd.Series({"A": 0.10, "B": 0.20})  # A is half as volatile as B
    cfg = PortfolioConfig(top_n_countries=2, max_country_weight=0.90)  # cap high enough to not bind

    weights = construct_country_weights(scores, predicted, vol, cfg)
    # inverse-vol: raw weights 1/0.10=10, 1/0.20=5 -> normalized 2/3, 1/3
    assert weights["A"] == pytest.approx(2.0 / 3.0, abs=1e-6)
    assert weights["B"] == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_construct_country_weights_never_exceeds_cap():
    scores = pd.Series({"A": 0.9})
    predicted = pd.Series({"A": 0.02})
    vol = pd.Series({"A": 0.10})
    cfg = PortfolioConfig(top_n_countries=3, max_country_weight=0.40)

    weights = construct_country_weights(scores, predicted, vol, cfg)
    non_cash = weights.drop(cfg.cash_key, errors="ignore")
    assert (non_cash <= cfg.max_country_weight + 1e-9).all()
    assert weights.sum() == pytest.approx(1.0)
