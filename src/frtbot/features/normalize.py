"""Cross-sectional winsorization and percentile ranking (SPEC.md section 6.4).

At the country-level vertical slice the "cross-section" is the set of five
markets on a given date (not stocks within a country - that is M3). Every
transform here operates strictly row-wise (same-date) or on trailing windows,
never using information from a later date, per SPEC.md: "winsorized using
trailing or same-date cross-sectional information only."
"""

from __future__ import annotations

import pandas as pd


def winsorize_cross_section(df: pd.DataFrame, limits: tuple[float, float] = (0.05, 0.05)) -> pd.DataFrame:
    """Clip each row to its own [lower_q, upper_q] quantiles across columns (markets)."""
    lower_q, upper_q = limits

    def _winsorize_row(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        if len(valid) < 2:
            return row
        lo = valid.quantile(lower_q)
        hi = valid.quantile(1.0 - upper_q)
        return row.clip(lower=lo, upper=hi)

    return df.apply(_winsorize_row, axis=1)


def cross_sectional_percentile_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Rank each row's non-NaN values to a [0, 1] percentile, NaN-safe per row."""
    return df.rank(axis=1, pct=True, na_option="keep")
