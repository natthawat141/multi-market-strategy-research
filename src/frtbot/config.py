"""Configuration schema and loaders for FRTBOT.

`SPEC.md` section 2 requires every instrument mapping to live in configuration,
not in model code, and to record provider, provider symbol, market, currency,
timezone, exchange calendar, price type, adjustment type, first valid date,
and last successful refresh. This module defines that schema (`MarketEntry`,
`FXEntry`) plus the research-run configuration (`ResearchConfig`) and loaders
that fail loudly on ambiguous or malformed configuration, per `AGENTS.md`.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

DataMode = Literal["stock", "proxy", "disabled"]
PriceType = Literal["close", "adjusted_close", "total_return"]
AdjustmentType = Literal[
    "split_dividend_adjusted",
    "split_adjusted",
    "unadjusted",
    "total_return_index",
]
CostScenario = Literal["zero", "optimistic", "base", "severe"]
InstrumentClass = Literal["country_proxy", "developed_stock", "emerging_stock"]


class FXEntry(BaseModel):
    """A daily FX series converting `base_currency` into `quote_currency`."""

    pair: str = Field(..., description='e.g. "USDTHB"')
    base_currency: str
    quote_currency: str
    provider: str
    provider_symbol: str
    timezone: str
    invert: bool = Field(
        False,
        description=(
            "Set when the provider only offers the inverse quote (e.g. yfinance has "
            "no direct CNYTHB=X but does have THBCNY=X); the fetched rate is then "
            "inverted (1/rate) to reach the declared base->quote convention."
        ),
    )
    first_valid_date: date | None = None
    last_refresh: datetime | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _check_currencies(self) -> FXEntry:
        if len(self.base_currency) != 3 or len(self.quote_currency) != 3:
            raise ValueError(
                f"FX pair {self.pair!r} must use 3-letter ISO currency codes, "
                f"got base={self.base_currency!r} quote={self.quote_currency!r}"
            )
        if self.pair != f"{self.base_currency}{self.quote_currency}":
            raise ValueError(
                f"FX pair {self.pair!r} does not match base/quote "
                f"{self.base_currency}{self.quote_currency}"
            )
        return self


class MarketEntry(BaseModel):
    """One country-level market proxy mapping (SPEC.md section 2)."""

    key: str = Field(..., description='Short internal code, e.g. "US", "TH"')
    name: str
    mode: DataMode
    currency: str
    timezone: str
    exchange_calendar: str = Field(..., description="pandas_market_calendars code, e.g. XNYS")
    provider: str
    provider_symbol: str
    price_type: PriceType
    adjustment_type: AdjustmentType
    instrument_class: InstrumentClass = "country_proxy"
    fx_pair: str | None = Field(
        None, description="FX pair key into FXConfig.fx, or null if currency == base_currency"
    )
    first_valid_date: date | None = None
    last_refresh: datetime | None = None
    disabled_reason: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> MarketEntry:
        if len(self.currency) != 3:
            raise ValueError(f"Market {self.key!r} has non-ISO currency {self.currency!r}")
        if self.mode == "disabled" and not self.disabled_reason:
            raise ValueError(f"Market {self.key!r} is disabled but has no disabled_reason")
        if self.mode != "disabled" and not self.provider_symbol:
            raise ValueError(f"Market {self.key!r} mode={self.mode!r} requires provider_symbol")
        return self


class MarketsConfig(BaseModel):
    """Top-level `configs/markets*.yml` document."""

    base_currency: str = "THB"
    markets: list[MarketEntry]
    fx: list[FXEntry]

    @model_validator(mode="after")
    def _check_unique_and_resolvable(self) -> MarketsConfig:
        keys = [m.key for m in self.markets]
        duplicates = {k for k in keys if keys.count(k) > 1}
        if duplicates:
            raise ValueError(f"Duplicate market keys in config: {sorted(duplicates)}")

        fx_pairs = [fx.pair for fx in self.fx]
        dup_fx = {p for p in fx_pairs if fx_pairs.count(p) > 1}
        if dup_fx:
            raise ValueError(f"Duplicate FX pairs in config: {sorted(dup_fx)}")
        fx_by_pair = {fx.pair: fx for fx in self.fx}

        for m in self.markets:
            if m.currency == self.base_currency:
                if m.fx_pair is not None:
                    raise ValueError(
                        f"Market {m.key!r} currency already equals base currency "
                        f"{self.base_currency!r}; fx_pair must be null"
                    )
                continue
            if m.mode == "disabled":
                continue
            expected_pair = f"{m.currency}{self.base_currency}"
            if m.fx_pair != expected_pair:
                raise ValueError(
                    f"Market {m.key!r} currency {m.currency!r} requires fx_pair "
                    f"{expected_pair!r}, got {m.fx_pair!r}"
                )
            if m.fx_pair not in fx_by_pair:
                raise ValueError(
                    f"Market {m.key!r} references fx_pair {m.fx_pair!r} "
                    f"which is not defined in fx: [...]"
                )
        return self

    def by_key(self, key: str) -> MarketEntry:
        for m in self.markets:
            if m.key == key:
                return m
        raise KeyError(f"Unknown market key {key!r}")

    def enabled_markets(self) -> list[MarketEntry]:
        return [m for m in self.markets if m.mode != "disabled"]

    def fx_for(self, market: MarketEntry) -> FXEntry | None:
        if market.fx_pair is None:
            return None
        for fx in self.fx:
            if fx.pair == market.fx_pair:
                return fx
        raise KeyError(f"FX pair {market.fx_pair!r} for market {market.key!r} not found")


class CostAssumption(BaseModel):
    """Per-side bps cost, by scenario and instrument class (SPEC.md section 10)."""

    zero: float
    optimistic: float
    base: float
    severe: float

    def bps(self, scenario: CostScenario) -> float:
        return getattr(self, scenario)


class CostsConfig(BaseModel):
    country_proxy: CostAssumption
    developed_stock: CostAssumption
    emerging_stock: CostAssumption

    def bps_for(self, instrument_class: InstrumentClass, scenario: CostScenario) -> float:
        return getattr(self, instrument_class).bps(scenario)


class WalkForwardConfig(BaseModel):
    train_years: int = 8
    val_years: int = 2
    test_years: int = 1
    step_years: int = 1
    min_train_years: int = 3
    embargo_days: int = 21


class PortfolioConfig(BaseModel):
    max_country_weight: float = 0.40
    top_n_countries: int = 3
    vol_lookback_days: int = 63
    cash_key: str = "CASH_THB"


class EnsembleConfig(BaseModel):
    tree_weight: float = 0.40
    mlp_weight: float = 0.30
    trend_weight: float = 0.30

    @model_validator(mode="after")
    def _check_sums_to_one(self) -> EnsembleConfig:
        total = self.tree_weight + self.mlp_weight + self.trend_weight
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Ensemble weights must sum to 1.0, got {total}")
        return self


class ResearchConfig(BaseModel):
    """Top-level `configs/research.yml` document."""

    base_currency: str = "THB"
    horizon_trading_days: int = 21
    cash_annual_rate: float = Field(
        0.0125,
        description=(
            "Documented modeling assumption for the THB cash-proxy return, not a "
            "downloaded market series. See README data-sources section."
        ),
    )
    seed: int = 42
    walk_forward: WalkForwardConfig = WalkForwardConfig()
    portfolio: PortfolioConfig = PortfolioConfig()
    ensemble: EnsembleConfig = EnsembleConfig()
    costs: CostsConfig


def load_markets_config(path: str | Path) -> MarketsConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Markets config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return MarketsConfig.model_validate(raw)


def load_research_config(path: str | Path) -> ResearchConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Research config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ResearchConfig.model_validate(raw)
