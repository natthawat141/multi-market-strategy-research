from __future__ import annotations

import pandas as pd
import pytest

from frtbot.data.fx import (
    align_fx_to_index,
    identity_fx_series,
    local_fx_return,
    thb_return,
    to_thb_price,
)
from frtbot.data.schema import SchemaError


def test_to_thb_price_is_local_price_times_rate():
    dates = pd.bdate_range("2020-01-01", periods=5)
    local_price = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=dates)
    fx_rate = pd.Series([35.0, 35.5, 36.0, 36.5, 37.0], index=dates)
    thb_price = to_thb_price(local_price, fx_rate)
    expected = local_price * fx_rate
    pd.testing.assert_series_equal(thb_price, expected, check_names=False)


def test_identity_fx_series_is_all_ones():
    dates = pd.bdate_range("2020-01-01", periods=5)
    fx = identity_fx_series(dates)
    assert (fx == 1.0).all()


def test_align_fx_forward_fills_short_gaps():
    fx_dates = pd.bdate_range("2020-01-01", periods=5)  # Wed 1/1 .. Tue 1/7 (business days)
    fx_rate = pd.Series([35.0, 35.1, 35.2, 35.3, 35.4], index=fx_dates)

    saturday_gap = fx_dates[2] + pd.Timedelta(days=1)  # Sat 1/4: no FX quote that day
    target_dates = pd.DatetimeIndex([fx_dates[0], saturday_gap, fx_dates[2]])
    aligned = align_fx_to_index(fx_rate, target_dates, max_ffill_days=5)
    assert aligned.iloc[1] == fx_rate.iloc[2]  # forward-filled from the prior Friday close
    assert aligned.iloc[2] == fx_rate.iloc[2]  # exact match


def test_align_fx_raises_on_stale_gap():
    fx_dates = pd.bdate_range("2020-01-01", periods=3)
    fx_rate = pd.Series([35.0, 35.1, 35.2], index=fx_dates)
    far_future = fx_dates[-1] + pd.Timedelta(days=30)
    target_dates = pd.DatetimeIndex([far_future])

    with pytest.raises(SchemaError):
        align_fx_to_index(fx_rate, target_dates, max_ffill_days=5)


def test_align_fx_raises_when_target_precedes_all_fx_data():
    fx_dates = pd.bdate_range("2020-06-01", periods=3)
    fx_rate = pd.Series([35.0, 35.1, 35.2], index=fx_dates)
    target_dates = pd.DatetimeIndex([pd.Timestamp("2020-01-01")])

    with pytest.raises(SchemaError):
        align_fx_to_index(fx_rate, target_dates, max_ffill_days=5)


def test_thb_return_matches_pct_change_of_thb_price():
    dates = pd.bdate_range("2020-01-01", periods=5)
    price_thb = pd.Series([100.0, 105.0, 110.0, 108.0, 112.0], index=dates)
    ret = thb_return(price_thb)
    expected = price_thb.pct_change()
    pd.testing.assert_series_equal(ret, expected, check_names=False)


def test_local_fx_return_positive_when_local_currency_strengthens():
    dates = pd.bdate_range("2020-01-01", periods=3)
    fx_rate = pd.Series([35.0, 36.0, 37.0], index=dates)  # more THB per unit local currency
    ret = local_fx_return(fx_rate)
    assert ret.dropna().gt(0).all()


def test_thb_return_decomposes_into_local_and_fx_return_approximately():
    dates = pd.bdate_range("2020-01-01", periods=2)
    local_price = pd.Series([100.0, 102.0], index=dates)  # +2% local
    fx_rate = pd.Series([35.0, 35.7], index=dates)  # +2% FX
    thb_price = to_thb_price(local_price, fx_rate)
    total_ret = thb_return(thb_price).dropna().iloc[0]
    local_ret = local_price.pct_change().dropna().iloc[0]
    fx_ret = local_fx_return(fx_rate).dropna().iloc[0]
    # (1+total) = (1+local)*(1+fx) exactly; first-order sum is a close approximation.
    assert total_ret == pytest.approx((1 + local_ret) * (1 + fx_ret) - 1, abs=1e-9)
