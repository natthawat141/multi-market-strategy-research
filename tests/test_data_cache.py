from __future__ import annotations

from datetime import date

from frtbot.reporting.data_audit import build_data_audit


def test_fixture_provider_output_is_tagged_synthetic(markets_config, data_cache, fixture_provider):
    m = markets_config.by_key("US")
    df, meta = data_cache.get_or_fetch(
        m.key, m.provider_symbol, "ohlcv", fixture_provider, date(2016, 1, 4), date(2018, 1, 1)
    )
    assert meta["source"] == "SYNTHETIC"
    assert df.attrs["source"] == "SYNTHETIC"


def test_raw_download_is_immutable_across_repeated_fetches(markets_config, data_cache, fixture_provider):
    m = markets_config.by_key("US")
    data_cache.get_or_fetch(m.key, m.provider_symbol, "ohlcv", fixture_provider, date(2016, 1, 4), date(2018, 1, 1))
    raw_files_first = sorted((data_cache.raw_dir).rglob("*.parquet"))

    # A second fetch on the same retrieval date must not create a second raw file.
    data_cache.get_or_fetch(
        m.key, m.provider_symbol, "ohlcv", fixture_provider, date(2016, 1, 4), date(2018, 1, 1), force_refresh=True
    )
    raw_files_second = sorted((data_cache.raw_dir).rglob("*.parquet"))
    assert raw_files_first == raw_files_second
    assert len(raw_files_second) == 1


def test_normalized_cache_is_reused_without_force_refresh(markets_config, data_cache, fixture_provider):
    m = markets_config.by_key("US")
    df1, meta1 = data_cache.get_or_fetch(
        m.key, m.provider_symbol, "ohlcv", fixture_provider, date(2016, 1, 4), date(2018, 1, 1)
    )
    df2, meta2 = data_cache.get_or_fetch(
        m.key, m.provider_symbol, "ohlcv", fixture_provider, date(2016, 1, 4), date(2018, 1, 1)
    )
    assert meta1["last_refresh"] == meta2["last_refresh"]  # not re-fetched


def test_data_audit_labels_disabled_market_without_touching_cache(markets_config, data_cache):
    disabled = markets_config.by_key("US").model_copy(update={"mode": "disabled", "disabled_reason": "test"})
    markets_config.markets[0] = disabled  # only mutate our local copy's list
    audit = build_data_audit(markets_config, data_cache)
    row = audit[audit["key"] == "US"].iloc[0]
    assert row["status"] == "disabled"


def test_data_audit_labels_uncached_market_as_missing(markets_config, data_cache):
    audit = build_data_audit(markets_config, data_cache)
    assert (audit["status"] == "missing").all()


def test_data_audit_price_anomaly_column_is_always_a_list(markets_config, data_cache, fixture_provider):
    """Regression test: audit_fx_entry previously omitted price_anomaly_dates, leaving FX
    rows NaN (a float) instead of [] and breaking any `.apply(len)` over the whole column."""
    for m in markets_config.markets:
        data_cache.get_or_fetch(m.key, m.provider_symbol, "ohlcv", fixture_provider, date(2016, 1, 4), date(2018, 1, 1))
    for fx in markets_config.fx:
        data_cache.get_or_fetch(fx.pair, fx.provider_symbol, "fx", fixture_provider, date(2016, 1, 4), date(2018, 1, 1))

    audit = build_data_audit(markets_config, data_cache)
    assert audit["price_anomaly_dates"].apply(lambda v: isinstance(v, list)).all()
    assert (audit["price_anomaly_dates"].apply(len) == 0).all()  # synthetic fixtures are smooth
