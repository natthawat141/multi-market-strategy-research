from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from frtbot.features.price_trend import (
    build_price_trend_features,
    donchian,
    momentum_12_1,
    rsi,
    sma_distance,
)
from frtbot.features.risk_liquidity import build_risk_liquidity_features, max_drawdown


def _uptrend_close(n: int = 400) -> pd.Series:
    dates = pd.bdate_range("2018-01-01", periods=n)
    return pd.Series(100.0 * (1.005 ** np.arange(n)), index=dates)


def _downtrend_close(n: int = 400) -> pd.Series:
    dates = pd.bdate_range("2018-01-01", periods=n)
    return pd.Series(100.0 * (0.995 ** np.arange(n)), index=dates)


def test_dist_sma_positive_in_uptrend():
    close = _uptrend_close()
    dist = sma_distance(close, windows=(50,))["dist_sma_50"]
    assert (dist.dropna().tail(50) > 0).all()


def test_dist_sma_negative_in_downtrend():
    close = _downtrend_close()
    dist = sma_distance(close, windows=(50,))["dist_sma_50"]
    assert (dist.dropna().tail(50) < 0).all()


def test_momentum_12_1_positive_in_sustained_uptrend():
    close = _uptrend_close()
    mom = momentum_12_1(close)
    assert mom.dropna().iloc[-1] > 0


def test_rsi_above_50_in_uptrend_and_bounded():
    close = _uptrend_close()
    r = rsi(close, window=14).dropna()
    assert (r.tail(50) > 50).all()
    assert (r <= 100).all() and (r >= 0).all()


def test_donchian_position_bounded_zero_one():
    close = _uptrend_close()
    high = close * 1.001
    low = close * 0.999
    pos = donchian(high, low, close, window=20)["donchian_20_position"].dropna()
    assert (pos >= -1e-9).all() and (pos <= 1.0 + 1e-9).all()


def test_max_drawdown_is_non_positive():
    close = _uptrend_close()
    mdd = max_drawdown(close, windows=(63,))["max_drawdown_63d"].dropna()
    assert (mdd <= 1e-9).all()


def test_max_drawdown_detects_known_drop():
    dates = pd.bdate_range("2018-01-01", periods=100)
    close = pd.Series(100.0, index=dates)
    close.iloc[50:] = 80.0  # sharp 20% drop, held afterwards
    mdd = max_drawdown(close, windows=(63,))["max_drawdown_63d"]
    assert mdd.iloc[-1] == pytest.approx(-0.20, abs=1e-6)


def test_price_trend_features_no_leakage_on_truncation():
    """Feature value at date t must not change when future rows are appended."""
    close = _uptrend_close(500)
    high, low = close * 1.001, close * 0.999

    cutoff = close.index[300]
    full = build_price_trend_features(close, high, low)
    truncated = build_price_trend_features(close.loc[:cutoff], high.loc[:cutoff], low.loc[:cutoff])

    pd.testing.assert_frame_equal(
        full.loc[:cutoff], truncated.loc[:cutoff], check_exact=False, rtol=1e-9
    )


def test_risk_liquidity_features_no_leakage_on_truncation():
    close = _uptrend_close(500)
    volume = pd.Series(1_000_000.0, index=close.index)
    daily_return = close.pct_change()

    cutoff = close.index[300]
    full = build_risk_liquidity_features(close, volume, daily_return)
    truncated = build_risk_liquidity_features(
        close.loc[:cutoff], volume.loc[:cutoff], daily_return.loc[:cutoff]
    )

    pd.testing.assert_frame_equal(
        full.loc[:cutoff], truncated.loc[:cutoff], check_exact=False, rtol=1e-9
    )


def test_full_feature_panel_has_no_nan_after_warmup(feature_panel):
    for key, frame in feature_panel.items():
        tail = frame.tail(100).drop(columns=["market"])
        assert tail.isna().sum().sum() == 0, f"{key} has unexpected NaN after warm-up"
