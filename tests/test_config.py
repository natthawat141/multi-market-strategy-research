from __future__ import annotations

import pytest
from pydantic import ValidationError

from frtbot.config import (
    EnsembleConfig,
    MarketEntry,
    MarketsConfig,
    load_markets_config,
    load_research_config,
)

FIXTURES_ROOT = __import__("pathlib").Path(__file__).resolve().parent / "fixtures"


def _entry(**overrides) -> dict:
    base = dict(
        key="US", name="US", mode="proxy", currency="USD", timezone="America/New_York",
        exchange_calendar="XNYS", provider="fixture", provider_symbol="US",
        price_type="adjusted_close", adjustment_type="split_dividend_adjusted",
        instrument_class="country_proxy", fx_pair="USDTHB",
    )
    base.update(overrides)
    return base


def _fx(**overrides) -> dict:
    base = dict(
        pair="USDTHB", base_currency="USD", quote_currency="THB",
        provider="fixture", provider_symbol="USDTHB", timezone="America/New_York",
    )
    base.update(overrides)
    return base


def test_example_configs_load_and_validate():
    mc = load_markets_config(FIXTURES_ROOT / "markets.fixture.yml")
    assert {m.key for m in mc.markets} == {"US", "EU", "JP", "CN", "TH"}
    rc = load_research_config("configs/research.yml")
    assert rc.ensemble.tree_weight + rc.ensemble.mlp_weight + rc.ensemble.trend_weight == pytest.approx(1.0)


def test_duplicate_market_keys_raise():
    with pytest.raises(ValidationError):
        MarketsConfig(
            base_currency="THB",
            markets=[MarketEntry(**_entry()), MarketEntry(**_entry())],
            fx=[_fx()],
        )


def test_disabled_market_requires_reason():
    with pytest.raises(ValidationError):
        MarketEntry(**_entry(mode="disabled", disabled_reason=None))


def test_market_currency_requires_matching_fx_pair():
    with pytest.raises(ValidationError):
        MarketsConfig(
            base_currency="THB",
            markets=[MarketEntry(**_entry(fx_pair="EURTHB"))],  # USD market but EURTHB pair
            fx=[_fx()],
        )


def test_market_referencing_undefined_fx_pair_raises():
    with pytest.raises(ValidationError):
        MarketsConfig(
            base_currency="THB",
            markets=[MarketEntry(**_entry())],
            fx=[],  # USDTHB never defined
        )


def test_base_currency_market_must_not_declare_fx_pair():
    with pytest.raises(ValidationError):
        MarketsConfig(
            base_currency="THB",
            markets=[MarketEntry(**_entry(key="TH", currency="THB", fx_pair="USDTHB"))],
            fx=[_fx()],
        )


def test_ensemble_weights_must_sum_to_one():
    with pytest.raises(ValidationError):
        EnsembleConfig(tree_weight=0.5, mlp_weight=0.3, trend_weight=0.3)
