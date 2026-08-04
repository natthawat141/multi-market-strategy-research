from __future__ import annotations

import pandas as pd
import pytest

from frtbot.data.schema import SchemaError, validate_fx, validate_ohlcv


def _valid_ohlcv(n: int = 5) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n, name="date")
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "adj_close": 100.5,
            "volume": 1_000_000.0,
        },
        index=dates,
    )


def test_valid_ohlcv_passes():
    df = _valid_ohlcv()
    out = validate_ohlcv(df, "test")
    assert len(out) == len(df)


def test_duplicate_timestamps_raise():
    df = _valid_ohlcv()
    dup = pd.concat([df, df.iloc[[0]]])
    with pytest.raises(SchemaError):
        validate_ohlcv(dup, "test")


def test_non_monotonic_timestamps_raise():
    df = _valid_ohlcv()
    shuffled = df.iloc[::-1]
    with pytest.raises(SchemaError):
        validate_ohlcv(shuffled, "test")


def test_negative_or_zero_price_raises():
    df = _valid_ohlcv()
    df.iloc[2, df.columns.get_loc("close")] = 0.0
    with pytest.raises(SchemaError):
        validate_ohlcv(df, "test")


def test_high_less_than_low_raises():
    df = _valid_ohlcv()
    df.iloc[1, df.columns.get_loc("high")] = 50.0  # below low=99
    with pytest.raises(SchemaError):
        validate_ohlcv(df, "test")


def test_nan_price_raises():
    df = _valid_ohlcv()
    df.iloc[0, df.columns.get_loc("close")] = float("nan")
    with pytest.raises(SchemaError):
        validate_ohlcv(df, "test")


def test_missing_column_raises():
    df = _valid_ohlcv().drop(columns=["volume"])
    with pytest.raises(SchemaError):
        validate_ohlcv(df, "test")


def test_timezone_aware_index_raises():
    df = _valid_ohlcv()
    df.index = df.index.tz_localize("UTC")
    with pytest.raises(SchemaError):
        validate_ohlcv(df, "test")


def test_valid_fx_passes():
    dates = pd.bdate_range("2020-01-01", periods=5, name="date")
    df = pd.DataFrame({"rate": 35.0}, index=dates)
    out = validate_fx(df, "test")
    assert len(out) == 5


def test_fx_non_positive_rate_raises():
    dates = pd.bdate_range("2020-01-01", periods=5, name="date")
    df = pd.DataFrame({"rate": 35.0}, index=dates)
    df.iloc[0, 0] = -1.0
    with pytest.raises(SchemaError):
        validate_fx(df, "test")
