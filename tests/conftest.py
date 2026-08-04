from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from frtbot.config import MarketsConfig, load_markets_config
from frtbot.data.cache import DataCache
from frtbot.data.fx import align_fx_to_index, identity_fx_series
from frtbot.data.providers import FixtureProvider, get_provider
from frtbot.features.build import MarketSeries, build_country_feature_panel

FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures"
FIXTURE_START = date(2016, 1, 4)
FIXTURE_END = date(2020, 5, 29)


@pytest.fixture
def markets_config() -> MarketsConfig:
    return load_markets_config(FIXTURES_ROOT / "markets.fixture.yml")


@pytest.fixture
def fixture_provider() -> FixtureProvider:
    return get_provider("fixture")


@pytest.fixture
def data_cache(tmp_path: Path) -> DataCache:
    return DataCache(root=tmp_path / "cache")


@pytest.fixture
def market_series(markets_config, data_cache, fixture_provider) -> dict[str, MarketSeries]:
    out = {}
    for m in markets_config.markets:
        df, _ = data_cache.get_or_fetch(
            m.key, m.provider_symbol, "ohlcv", fixture_provider, FIXTURE_START, FIXTURE_END
        )
        if m.fx_pair is None:
            fx_aligned = identity_fx_series(df.index)
        else:
            fx_entry = markets_config.fx_for(m)
            fx_df, _ = data_cache.get_or_fetch(
                fx_entry.pair, fx_entry.provider_symbol, "fx", fixture_provider, FIXTURE_START, FIXTURE_END
            )
            fx_aligned = align_fx_to_index(fx_df["rate"], df.index)
        out[m.key] = MarketSeries(key=m.key, ohlcv=df, fx_rate_aligned=fx_aligned)
    return out


@pytest.fixture
def feature_panel(market_series) -> dict[str, pd.DataFrame]:
    return build_country_feature_panel(market_series)
