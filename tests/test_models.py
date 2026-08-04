from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from frtbot.config import EnsembleConfig
from frtbot.models.base import same_date_percentile_rank
from frtbot.models.baselines import EqualWeightModel, RuleMomentumModel
from frtbot.models.ensemble import combine_ensemble
from frtbot.models.trend import TransparentTrendModel


def _toy_long_df(n_dates: int = 5) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    markets = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product([dates, markets], names=["date", "market"])
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "dist_sma_50": rng.normal(size=len(idx)),
            "dist_sma_200": rng.normal(size=len(idx)),
            "mom_12_1": rng.normal(size=len(idx)),
            "donchian_20_position": rng.uniform(size=len(idx)),
            "rsi_14": rng.uniform(20, 80, size=len(idx)),
            "thb_ret_63d": rng.normal(size=len(idx)) * 0.05,
            "rel_mom_rank": rng.uniform(size=len(idx)),
        },
        index=idx,
    )
    return df


def test_same_date_percentile_rank_is_within_date_only():
    scores = pd.Series(
        [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        index=pd.MultiIndex.from_tuples(
            [("d1", "A"), ("d1", "B"), ("d1", "C"), ("d2", "A"), ("d2", "B"), ("d2", "C")],
            names=["date", "market"],
        ),
    )
    ranks = same_date_percentile_rank(scores)
    # Highest score within each date gets rank 1.0, regardless of absolute scale across dates.
    assert ranks.loc[("d1", "C")] == 1.0
    assert ranks.loc[("d2", "C")] == 1.0
    assert ranks.loc[("d1", "A")] == pytest.approx(1 / 3)


def test_equal_weight_model_gives_identical_positive_score():
    df = _toy_long_df()
    scores = EqualWeightModel().predict(df)
    assert (scores == scores.iloc[0]).all()
    assert (scores > 0).all()


def test_rule_momentum_score_bounded_zero_one():
    df = _toy_long_df()
    scores = RuleMomentumModel().predict(df)
    assert (scores >= 0).all() and (scores <= 1).all()


def test_transparent_trend_score_is_stateless_and_bounded():
    df = _toy_long_df()
    model = TransparentTrendModel()
    scores_before_fit = model.predict(df)
    model.fit(df)  # no-op; must not change behavior
    scores_after_fit = model.predict(df)
    pd.testing.assert_series_equal(scores_before_fit, scores_after_fit)
    assert (scores_after_fit >= 0).all() and (scores_after_fit <= 1).all()


def test_ensemble_combines_with_frozen_weights():
    dates = pd.bdate_range("2020-01-01", periods=1)
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "market"])
    scores = {
        "tree": pd.Series([1.0, 2.0], index=idx),
        "mlp": pd.Series([2.0, 1.0], index=idx),
        "transparent_trend": pd.Series([1.0, 2.0], index=idx),
    }
    cfg = EnsembleConfig(tree_weight=0.4, mlp_weight=0.3, trend_weight=0.3)
    result = combine_ensemble(scores, cfg)
    assert result.excluded == []
    assert result.weights_used == {"tree": 0.4, "mlp": 0.3, "transparent_trend": 0.3}
    # B ranks #1 on tree(0.4) and trend(0.3) = 0.7 weight at rank 1.0; A ranks #1 only on mlp(0.3).
    assert result.score.loc[(dates[0], "B")] > result.score.loc[(dates[0], "A")]


def test_ensemble_excludes_and_renormalizes_missing_model():
    dates = pd.bdate_range("2020-01-01", periods=1)
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "market"])
    scores = {
        "tree": pd.Series([1.0, 2.0], index=idx),
        "transparent_trend": pd.Series([1.0, 2.0], index=idx),
        # "mlp" deliberately missing (e.g. it failed validation)
    }
    cfg = EnsembleConfig(tree_weight=0.4, mlp_weight=0.3, trend_weight=0.3)
    result = combine_ensemble(scores, cfg)
    assert result.excluded == ["mlp"]
    assert sum(result.weights_used.values()) == pytest.approx(1.0)
    assert result.weights_used["tree"] == pytest.approx(0.4 / 0.7)
