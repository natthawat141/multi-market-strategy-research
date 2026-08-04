"""Risk and liquidity features (SPEC.md section 6.2).

Same leakage discipline as `price_trend.py`: every statistic at row `t` is a
function of a trailing window ending at `t`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
VOL_WINDOWS = (20, 63, 126)
DRAWDOWN_WINDOWS = (63, 252)


def realized_vol(
    daily_return: pd.Series, windows: tuple[int, ...] = VOL_WINDOWS
) -> pd.DataFrame:
    out = {}
    for w in windows:
        out[f"vol_{w}d"] = daily_return.rolling(w).std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return pd.DataFrame(out, index=daily_return.index)


def downside_vol(daily_return: pd.Series, window: int = 63) -> pd.Series:
    downside = daily_return.clip(upper=0.0)
    return (downside.rolling(window).std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)).rename(
        f"downside_vol_{window}d"
    )


def _rolling_max_drawdown(close: pd.Series, window: int) -> pd.Series:
    def _mdd(x: np.ndarray) -> float:
        running_max = np.maximum.accumulate(x)
        drawdown = x / running_max - 1.0
        return float(drawdown.min())

    return close.rolling(window).apply(_mdd, raw=True)


def max_drawdown(close: pd.Series, windows: tuple[int, ...] = DRAWDOWN_WINDOWS) -> pd.DataFrame:
    out = {f"max_drawdown_{w}d": _rolling_max_drawdown(close, w) for w in windows}
    return pd.DataFrame(out, index=close.index)


def avg_traded_value_zscore(
    close: pd.Series, volume: pd.Series, window: int = 63
) -> pd.Series:
    traded_value = close * volume
    mean = traded_value.rolling(window).mean()
    std = traded_value.rolling(window).std(ddof=0).replace(0.0, np.nan)
    return ((traded_value - mean) / std).rename(f"traded_value_zscore_{window}d")


def volume_zscore(volume: pd.Series, window: int = 63) -> pd.Series:
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std(ddof=0).replace(0.0, np.nan)
    return ((volume - mean) / std).rename(f"volume_zscore_{window}d")


def zero_volume_stale_rate(
    close: pd.Series, volume: pd.Series, window: int = 63
) -> pd.Series:
    """Fraction of trailing `window` days with zero volume or an unchanged close."""
    stale = (volume <= 0) | (close.diff() == 0.0)
    return stale.rolling(window).mean().rename(f"stale_rate_{window}d")


def build_risk_liquidity_features(
    close: pd.Series, volume: pd.Series, daily_return: pd.Series
) -> pd.DataFrame:
    """Assemble every SPEC.md section 6.2 feature for one market proxy."""
    frames = [
        realized_vol(daily_return),
        downside_vol(daily_return).to_frame(),
        max_drawdown(close),
        avg_traded_value_zscore(close, volume).to_frame(),
        volume_zscore(volume).to_frame(),
        zero_volume_stale_rate(close, volume).to_frame(),
    ]
    return pd.concat(frames, axis=1)
