"""Assemble the long-format (date, market) modeling panel used by every model.

Feature warm-up (e.g. the first ~252 rows lack a valid `ret_252d`) and label
warm-down (the last `horizon` rows lack a forward label, SPEC.md 7.1) both
produce NaNs by construction. `drop_incomplete` removes them explicitly so
every model sees the same, clearly-defined training rows rather than each
silently handling NaNs differently.
"""

from __future__ import annotations

import pandas as pd

LABEL_COLUMNS = ("label_regression", "label_classification")


def build_country_dataset(
    feature_panel: dict[str, pd.DataFrame],
    regression_labels: dict[str, pd.Series],
    classification_labels: dict[str, pd.Series],
) -> pd.DataFrame:
    """Long-format panel indexed by (date, market) with feature + label columns."""
    frames = []
    for key, feat in feature_panel.items():
        f = feat.drop(columns=["market"]).copy()
        f["label_regression"] = regression_labels[key].reindex(f.index)
        f["label_classification"] = classification_labels[key].reindex(f.index)
        f["market"] = key
        f["date"] = f.index
        frames.append(f)
    long_df = pd.concat(frames, axis=0)
    long_df = long_df.set_index(["date", "market"]).sort_index()
    return long_df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in LABEL_COLUMNS]


def drop_incomplete(
    df: pd.DataFrame, feature_cols: list[str], label_col: str = "label_regression"
) -> pd.DataFrame:
    """Drop rows with any NaN feature or a NaN label (warm-up/warm-down rows)."""
    mask = df[feature_cols].notna().all(axis=1) & df[label_col].notna()
    return df.loc[mask]


def select_dates(df: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Slice a (date, market)-indexed frame to a set of dates on the `date` level."""
    return df.loc[df.index.get_level_values("date").isin(dates)]
