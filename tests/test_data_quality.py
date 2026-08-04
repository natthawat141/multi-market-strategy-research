from __future__ import annotations

import numpy as np
import pandas as pd

from frtbot.data.quality import detect_extreme_return_days


def test_no_anomalies_in_normal_series():
    dates = pd.bdate_range("2020-01-01", periods=100)
    close = pd.Series(100.0 * (1.001 ** np.arange(100)), index=dates)  # smooth uptrend
    assert len(detect_extreme_return_days(close)) == 0


def test_detects_split_style_glitch():
    dates = pd.bdate_range("2020-01-01", periods=10)
    close = pd.Series([100.0, 101.0, 102.0, 10.0, 9.9, 103.0, 104.0, 105.0, 106.0, 107.0], index=dates)
    anomalies = detect_extreme_return_days(close)
    # The crash-in and jump-back-out days both exceed the threshold.
    assert dates[3] in anomalies
    assert dates[5] in anomalies


def test_threshold_is_configurable():
    dates = pd.bdate_range("2020-01-01", periods=5)
    close = pd.Series([100.0, 120.0, 121.0, 122.0, 123.0], index=dates)  # one +20% day
    assert len(detect_extreme_return_days(close, max_abs_daily_return=0.40)) == 0
    assert len(detect_extreme_return_days(close, max_abs_daily_return=0.10)) == 1
