"""Generate the small deterministic SYNTHETIC fixture dataset used by tests.

Per SPEC.md section 5.3 ("A small deterministic fixture dataset is committed
only for tests") and section 3 ("all mock outputs must be labelled
SYNTHETIC"). This script is deterministic (fixed per-series seeds) and is run
once; its parquet output is committed under tests/fixtures/synthetic/. Re-run
it only if the fixture shape needs to change:

    .venv/Scripts/python.exe tests/fixtures/generate_synthetic_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES_DIR = Path(__file__).resolve().parent / "synthetic"
N_DAYS = 1150  # ~4.5 trading years: enough for 252d features + a small walk-forward fold.
START = "2016-01-04"

# (key, seed, annual_drift, annual_vol) - deliberately differentiated so
# cross-market relative-momentum and trend features are non-degenerate.
MARKETS = [
    ("US", 101, 0.09, 0.16),
    ("EU", 102, 0.04, 0.18),
    ("JP", 103, 0.06, 0.17),
    ("CN", 104, 0.02, 0.26),
    ("TH", 105, 0.05, 0.15),
]

# (pair, seed, level, annual_vol)
FX = [
    ("USDTHB", 201, 34.5, 0.06),
    ("EURTHB", 202, 37.5, 0.07),
    ("JPYTHB", 203, 0.245, 0.07),
    ("CNYTHB", 204, 4.85, 0.06),
]


def _gbm_close(seed: int, drift: float, vol: float, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    shocks = rng.normal((drift - 0.5 * vol**2) * dt, vol * np.sqrt(dt), size=n)
    log_price = np.cumsum(shocks)
    return 100.0 * np.exp(log_price)


def make_ohlcv(key: str, seed: int, drift: float, vol: float) -> pd.DataFrame:
    dates = pd.bdate_range(START, periods=N_DAYS, name="date")
    close = _gbm_close(seed, drift, vol, N_DAYS)
    rng = np.random.default_rng(seed + 1)

    open_gap = rng.normal(0, 0.003, N_DAYS)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    open_ = prev_close * (1 + open_gap)

    intraday = np.abs(rng.normal(0, 0.006, N_DAYS))
    high = np.maximum(open_, close) * (1 + intraday)
    low = np.minimum(open_, close) * (1 - intraday)

    volume = rng.lognormal(mean=14.0, sigma=0.3, size=N_DAYS).round()

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        },
        index=dates,
    )
    df.attrs["source"] = "SYNTHETIC"
    return df


def make_fx(pair: str, seed: int, level: float, vol: float) -> pd.DataFrame:
    dates = pd.bdate_range(START, periods=N_DAYS, name="date")
    rate = _gbm_close(seed, 0.0, vol, N_DAYS) * (level / 100.0)
    df = pd.DataFrame({"rate": rate}, index=dates)
    df.attrs["source"] = "SYNTHETIC"
    return df


def main() -> None:
    ohlcv_dir = FIXTURES_DIR / "ohlcv"
    fx_dir = FIXTURES_DIR / "fx"
    ohlcv_dir.mkdir(parents=True, exist_ok=True)
    fx_dir.mkdir(parents=True, exist_ok=True)

    for key, seed, drift, vol in MARKETS:
        df = make_ohlcv(key, seed, drift, vol)
        df.to_parquet(ohlcv_dir / f"{key}.parquet")
        print(f"wrote {key}: {len(df)} rows, {df.index.min().date()}..{df.index.max().date()}")

    for pair, seed, level, vol in FX:
        df = make_fx(pair, seed, level, vol)
        df.to_parquet(fx_dir / f"{pair}.parquet")
        print(f"wrote {pair}: {len(df)} rows, level~{df['rate'].mean():.3f}")


if __name__ == "__main__":
    main()
