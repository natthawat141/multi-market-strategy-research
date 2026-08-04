"""Assemble the full country-level feature panel across all five markets.

Design choices (documented here since SPEC.md leaves them to the
implementation):

- Technical/trend and risk/liquidity features (SPEC.md 6.1, 6.2) are computed
  on each market's *local-currency* adjusted OHLCV, the conventional basis for
  trend indicators, so FX swings do not distort channel/SMA/RSI structure.
- THB-denominated returns (`thb_ret_{w}d`) are provided explicitly alongside,
  since the research question (SPEC.md section 4) is framed in THB excess
  return.
- Cross-market relative momentum rank (6.3) ranks *THB-denominated* 12-1
  momentum across markets, since it is meant to proxy relative attractiveness
  to a THB investor, not just local trend strength.
- Same-date cross-sectional comparison across the 5 markets assumes their
  daily closes are usable as of the same trading date, a documented
  simplification at the daily-close granularity of this country-level slice
  (see README limitations).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from frtbot.data.fx import local_fx_return, to_thb_price
from frtbot.features.market_context import (
    cross_market_relative_momentum_rank,
    global_equal_weight_return,
    global_risk_regime,
    rolling_corr_with_global,
)
from frtbot.features.price_trend import build_price_trend_features, momentum_12_1
from frtbot.features.risk_liquidity import build_risk_liquidity_features

THB_RETURN_WINDOWS = (21, 63, 252)


@dataclass
class MarketSeries:
    """One market's local-currency OHLCV plus its THB-per-local-currency-unit rate.

    `fx_rate_aligned` must already be aligned to `ohlcv.index` (see
    `frtbot.data.fx.align_fx_to_index` / `identity_fx_series`).
    """

    key: str
    ohlcv: pd.DataFrame
    fx_rate_aligned: pd.Series


def build_market_features(series: MarketSeries) -> pd.DataFrame:
    """SPEC.md 6.1 + 6.2 features plus THB returns and local-FX-return for one market."""
    close, high, low, volume = (
        series.ohlcv["close"],
        series.ohlcv["high"],
        series.ohlcv["low"],
        series.ohlcv["volume"],
    )
    daily_return = close.pct_change()

    trend = build_price_trend_features(close, high, low)
    risk = build_risk_liquidity_features(close, volume, daily_return)

    thb_close = to_thb_price(close, series.fx_rate_aligned)
    thb_ret = pd.DataFrame(
        {f"thb_ret_{w}d": thb_close.pct_change(w) for w in THB_RETURN_WINDOWS},
        index=close.index,
    )
    fx_ret = local_fx_return(series.fx_rate_aligned).to_frame("local_fx_return_vs_thb")

    out = pd.concat([trend, risk, thb_ret, fx_ret], axis=1)
    out.insert(0, "market", series.key)
    return out


def build_country_feature_panel(
    market_series: dict[str, MarketSeries], corr_window: int = 63
) -> dict[str, pd.DataFrame]:
    """Build every market's full feature frame, including cross-market context columns."""
    per_market = {key: build_market_features(s) for key, s in market_series.items()}

    thb_close_by_market = {
        key: to_thb_price(s.ohlcv["close"], s.fx_rate_aligned)
        for key, s in market_series.items()
    }
    thb_mom_by_market = {key: momentum_12_1(c) for key, c in thb_close_by_market.items()}
    rel_mom_rank = cross_market_relative_momentum_rank(thb_mom_by_market)

    thb_return_by_market = {key: c.pct_change() for key, c in thb_close_by_market.items()}
    global_ret = global_equal_weight_return(thb_return_by_market)
    regime = global_risk_regime(global_ret)
    corr = rolling_corr_with_global(thb_return_by_market, global_ret, window=corr_window)

    for key, frame in per_market.items():
        frame["rel_mom_rank"] = rel_mom_rank[f"{key}_rel_mom_rank"].reindex(frame.index)
        frame["global_risk_regime_z"] = regime.reindex(frame.index)
        frame["corr_global"] = corr[f"{key}_corr_global_{corr_window}d"].reindex(frame.index)

    return per_market
