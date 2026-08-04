"""Price and trend features (SPEC.md section 6.1).

Every function is a pure trailing-window transform of `close`/`high`/`low`:
the value at row `t` uses only data up to and including `t`, so nothing here
introduces same-bar leakage on its own. SPEC.md section 6 leakage protection
("tradable no earlier than the next valid local-market bar") is enforced by
the backtest execution layer, not by shifting feature values themselves -
what is "known" at close of `t` legitimately includes close of `t`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RETURN_WINDOWS = (5, 21, 63, 126, 252)
SMA_WINDOWS = (20, 50, 100, 200)
DONCHIAN_WINDOWS = (20, 55)


def returns(close: pd.Series, windows: tuple[int, ...] = RETURN_WINDOWS) -> pd.DataFrame:
    out = {f"ret_{w}d": close.pct_change(w) for w in windows}
    return pd.DataFrame(out, index=close.index)


def momentum_12_1(close: pd.Series) -> pd.Series:
    """Conventional 12-to-1-month momentum: return from t-252 to t-21 (skip the last month)."""
    return (close.shift(21) / close.shift(252) - 1.0).rename("mom_12_1")


def sma_distance(close: pd.Series, windows: tuple[int, ...] = SMA_WINDOWS) -> pd.DataFrame:
    out = {}
    for w in windows:
        sma = close.rolling(w).mean()
        out[f"dist_sma_{w}"] = close / sma - 1.0
    return pd.DataFrame(out, index=close.index)


def sma_slope(close: pd.Series, window: int, slope_lookback: int = 20) -> pd.Series:
    """Percent change of the SMA(`window`) itself over `slope_lookback` days."""
    sma = close.rolling(window).mean()
    return (sma / sma.shift(slope_lookback) - 1.0).rename(f"sma_{window}_slope")


def ema_diff_norm(close: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return ((ema_fast - ema_slow) / close).rename(f"ema_{fast}_{slow}_diff_norm")


def donchian(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.DataFrame:
    """Donchian channel position in [0, 1] plus a breakout-above-prior-channel flag."""
    highest = high.rolling(window).max()
    lowest = low.rolling(window).min()
    channel_range = (highest - lowest).replace(0.0, np.nan)
    position = (close - lowest) / channel_range
    breakout = (close > highest.shift(1)).astype(float)
    return pd.DataFrame(
        {
            f"donchian_{window}_position": position,
            f"donchian_{window}_breakout": breakout,
        },
        index=close.index,
    )


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.where(avg_loss != 0.0, 100.0)
    return out.rename(f"rsi_{window}")


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's ADX (average directional index)."""
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean() / atr
    minus_di = (
        100.0 * minus_dm.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean() / atr
    )
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean().rename(
        f"adx_{window}"
    )


def dist_from_52w_high(close: pd.Series, window: int = 252) -> pd.Series:
    rolling_high = close.rolling(window).max()
    return (close / rolling_high - 1.0).rename("dist_52w_high")


def build_price_trend_features(
    close: pd.Series, high: pd.Series, low: pd.Series
) -> pd.DataFrame:
    """Assemble every SPEC.md section 6.1 feature for one market proxy."""
    frames = [
        returns(close),
        momentum_12_1(close).to_frame(),
        sma_distance(close),
        sma_slope(close, 50).to_frame(),
        sma_slope(close, 200).to_frame(),
        ema_diff_norm(close).to_frame(),
        donchian(high, low, close, 20),
        donchian(high, low, close, 55),
        rsi(close, 14).to_frame(),
        adx(high, low, close, 14).to_frame(),
        dist_from_52w_high(close).to_frame(),
    ]
    return pd.concat(frames, axis=1)
