"""Data audit report (SPEC.md acceptance criteria #2, #3).

Classifies every configured market/FX series as `usable`, `proxy`,
`disabled`, `missing`, `stale`, or `synthetic`, and reports coverage, gaps,
staleness, currency, and adjustment metadata. Built entirely from already
locally-cached data + its metadata sidecar (`DataCache`), so it never needs
network access on its own.

Markets are additionally checked for implausible single-day price moves
(`price_anomaly_dates`, see `frtbot.data.quality`) - an orthogonal
plausibility flag layered on top of `status`, since a market can be
nominally `proxy`/reachable yet still contain a provider-side data defect
that should exclude it from a trusted run.
"""

from __future__ import annotations

import pandas as pd

from frtbot.config import FXEntry, MarketEntry, MarketsConfig
from frtbot.data.cache import DataCache
from frtbot.data.quality import detect_extreme_return_days

DEFAULT_STALE_THRESHOLD_DAYS = 10


def _coverage_stats(df: pd.DataFrame, meta: dict) -> dict:
    first = pd.Timestamp(meta["first_valid_date"])
    last = pd.Timestamp(meta["last_valid_date"])
    expected_bdays = len(pd.bdate_range(first, last)) if last >= first else 0
    coverage_ratio = (len(df) / expected_bdays) if expected_bdays else None
    gaps = df.index.to_series().diff().dt.days.dropna()
    max_gap_days = int(gaps.max()) if len(gaps) else None
    return {
        "first_valid_date": first.date(),
        "last_valid_date": last.date(),
        "row_count": int(meta["row_count"]),
        "business_day_coverage_ratio": coverage_ratio,
        "max_gap_days": max_gap_days,
    }


def audit_market_entry(
    entry: MarketEntry,
    cached: tuple[pd.DataFrame, dict] | None,
    as_of: pd.Timestamp,
    stale_threshold_days: int = DEFAULT_STALE_THRESHOLD_DAYS,
) -> dict:
    row: dict = {
        "key": entry.key,
        "name": entry.name,
        "kind": "market",
        "configured_mode": entry.mode,
        "currency": entry.currency,
        "provider": entry.provider,
        "provider_symbol": entry.provider_symbol,
        "price_type": entry.price_type,
        "adjustment_type": entry.adjustment_type,
        "disabled_reason": entry.disabled_reason,
        "source": None,
        "first_valid_date": None,
        "last_valid_date": None,
        "row_count": 0,
        "business_day_coverage_ratio": None,
        "max_gap_days": None,
        "stale_now": None,
        "price_anomaly_dates": [],
    }

    if entry.mode == "disabled":
        row["status"] = "disabled"
        return row

    if cached is None:
        row["status"] = "missing"
        return row

    df, meta = cached
    stats = _coverage_stats(df, meta)
    row.update(stats)
    row["source"] = meta.get("source", "UNKNOWN")
    stale_now = (as_of - pd.Timestamp(stats["last_valid_date"])).days > stale_threshold_days
    row["stale_now"] = stale_now
    row["price_anomaly_dates"] = [d.date() for d in detect_extreme_return_days(df["close"])]

    if row["source"] == "SYNTHETIC":
        row["status"] = "synthetic"
    elif stale_now:
        row["status"] = "stale"
    elif entry.mode == "stock":
        row["status"] = "usable"
    else:
        row["status"] = "proxy"
    return row


def audit_fx_entry(
    entry: FXEntry,
    cached: tuple[pd.DataFrame, dict] | None,
    as_of: pd.Timestamp,
    stale_threshold_days: int = DEFAULT_STALE_THRESHOLD_DAYS,
) -> dict:
    row: dict = {
        "key": entry.pair,
        "name": f"{entry.base_currency}->{entry.quote_currency} FX",
        "kind": "fx",
        "configured_mode": "proxy",
        "currency": entry.base_currency,
        "provider": entry.provider,
        "provider_symbol": entry.provider_symbol,
        "price_type": None,
        "adjustment_type": None,
        "disabled_reason": None,
        "source": None,
        "first_valid_date": None,
        "last_valid_date": None,
        "row_count": 0,
        "business_day_coverage_ratio": None,
        "max_gap_days": None,
        "stale_now": None,
        "price_anomaly_dates": [],
    }
    if cached is None:
        row["status"] = "missing"
        return row

    df, meta = cached
    stats = _coverage_stats(df, meta)
    row.update(stats)
    row["source"] = meta.get("source", "UNKNOWN")
    stale_now = (as_of - pd.Timestamp(stats["last_valid_date"])).days > stale_threshold_days
    row["stale_now"] = stale_now
    row["status"] = "synthetic" if row["source"] == "SYNTHETIC" else ("stale" if stale_now else "proxy")
    return row


def build_data_audit(
    markets_config: MarketsConfig,
    cache: DataCache,
    as_of: pd.Timestamp | None = None,
    stale_threshold_days: int = DEFAULT_STALE_THRESHOLD_DAYS,
) -> pd.DataFrame:
    """One row per configured market and FX pair; see module docstring for `status` values."""
    as_of = as_of or pd.Timestamp.now().normalize()
    rows = []
    for m in markets_config.markets:
        cached = cache.load_normalized(m.key, "ohlcv")
        rows.append(audit_market_entry(m, cached, as_of, stale_threshold_days))
    for fx in markets_config.fx:
        cached = cache.load_normalized(fx.pair, "fx")
        rows.append(audit_fx_entry(fx, cached, as_of, stale_threshold_days))
    return pd.DataFrame(rows)
