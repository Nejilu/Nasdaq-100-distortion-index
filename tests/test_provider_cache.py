from datetime import date

import pandas as pd

import provider_cache
from cached_providers import (
    CachedAcwiFloatWeightsProvider,
    CachedHoldingsProvider,
    provider_cache_key,
)
from nasdaq100_rebalance import SelectionResult
from provider_cache import ProviderCache, holdings_fingerprint
from qqq_holdings_provider import HttpCsvHoldingsProvider


def _selection() -> SelectionResult:
    securities = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "company_name": ["Alpha", "Beta"],
            "company_id": ["CIK:1", "CIK:2"],
            "security_type": ["Ordinary share", "Ordinary share"],
            "is_current": [True, True],
            "selected": [True, True],
        }
    )
    return SelectionResult(
        securities=securities,
        selected_tickers=("A", "B"),
        selected_company_ids=("CIK:1", "CIK:2"),
        additions=(),
        removals=(),
        status="test",
        source="fixture",
        as_of=date(2026, 7, 29).isoformat(),
        eligible_company_count=2,
        notes=("cached",),
    )


def test_market_data_cache_roundtrip(tmp_path):
    cache = ProviderCache(tmp_path / "providers.sqlite3")
    rows = pd.DataFrame(
        {
            "ticker": ["A"],
            "price": [10.0],
            "float_shares": [90.0],
            "shares_outstanding": [100.0],
            "market_cap": [1_000.0],
            "float_shares_status": ["reported"],
            "shares_outstanding_status": ["reported"],
            "market_data_error": [None],
            "price_fetched_at": [100.0],
            "fundamentals_fetched_at": [90.0],
            "fundamentals_attempted_at": [90.0],
        }
    )

    cache.upsert_market_data(rows)
    result = cache.get_market_data(["A", "MISSING"]).set_index("ticker")

    assert list(result.index) == ["A"]
    assert result.loc["A", "price"] == 10.0
    assert result.loc["A", "shares_outstanding"] == 100.0


def test_selection_cache_reports_fresh_and_stale_entries(tmp_path, monkeypatch):
    cache = ProviderCache(tmp_path / "providers.sqlite3")
    holdings = pd.DataFrame({"ticker": ["A", "B"]})
    fingerprint = holdings_fingerprint(holdings)
    monkeypatch.setattr(provider_cache.time, "time", lambda: 1_000.0)
    cache.save_selection("non_ucits", fingerprint, _selection())

    monkeypatch.setattr(provider_cache.time, "time", lambda: 1_030.0)
    fresh = cache.get_selection(
        "non_ucits",
        fingerprint,
        max_age_seconds=60,
    )
    assert fresh is not None
    assert fresh.is_fresh
    assert fresh.holdings_match
    assert fresh.selection.selected_tickers == ("A", "B")

    monkeypatch.setattr(provider_cache.time, "time", lambda: 1_120.0)
    stale = cache.get_selection(
        "non_ucits",
        fingerprint,
        max_age_seconds=60,
    )
    assert stale is not None
    assert not stale.is_fresh
    assert stale.age_seconds == 120.0


def test_selection_cache_detects_holdings_change(tmp_path):
    cache = ProviderCache(tmp_path / "providers.sqlite3")
    original = holdings_fingerprint(pd.DataFrame({"ticker": ["A", "B"]}))
    changed = holdings_fingerprint(pd.DataFrame({"ticker": ["A", "C"]}))
    cache.save_selection("non_ucits", original, _selection())

    result = cache.get_selection(
        "non_ucits",
        changed,
        max_age_seconds=86_400,
    )

    assert result is not None
    assert not result.holdings_match


def test_background_job_status_roundtrip(tmp_path):
    cache = ProviderCache(tmp_path / "providers.sqlite3")

    cache.set_job_status("job", "running")
    running = cache.get_job("job")
    cache.set_job_status("job", "complete", snapshot_id=42)
    complete = cache.get_job("job")

    assert running is not None
    assert running["status"] == "running"
    assert complete is not None
    assert complete["status"] == "complete"
    assert complete["snapshot_id"] == 42


def test_provider_frame_cache_preserves_data_attrs_and_metadata(
    tmp_path,
    monkeypatch,
):
    cache = ProviderCache(tmp_path / "providers.sqlite3")
    frame = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "actual_weight": [0.6, 0.4],
        }
    )
    frame.attrs["published_weight_total"] = 0.99
    monkeypatch.setattr(provider_cache.time, "time", lambda: 1_000.0)
    cache.save_provider_frame(
        "holdings",
        frame,
        source_name="issuer",
        reference_fund="ETF",
        holdings_as_of="2026-07-29",
        failures=("primary failed",),
    )

    monkeypatch.setattr(provider_cache.time, "time", lambda: 1_030.0)
    result = cache.get_provider_frame(
        "holdings",
        max_age_seconds=60,
    )

    assert result is not None
    assert result.is_fresh
    pd.testing.assert_frame_equal(
        result.frame,
        frame,
        check_exact=False,
        rtol=1e-14,
    )
    assert result.frame.attrs["published_weight_total"] == 0.99
    assert result.source_name == "issuer"
    assert result.reference_fund == "ETF"
    assert result.holdings_as_of == "2026-07-29"
    assert result.failures == ("primary failed",)


class _CountingHoldingsProvider:
    source_name = "live_issuer"
    reference_fund = "ETF"
    holdings_as_of = "2026-07-29"
    failures: tuple[str, ...] = ()

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    def get_holdings(self):
        self.calls += 1
        if self.error:
            raise self.error
        frame = pd.DataFrame(
            {
                "ticker": ["A"],
                "company_name": ["Alpha"],
                "actual_weight": [1.0],
            }
        )
        frame.attrs["holdings_as_of"] = self.holdings_as_of
        return frame


def test_cached_holdings_provider_avoids_live_call_when_fresh(tmp_path):
    cache = ProviderCache(tmp_path / "providers.sqlite3")
    first = _CountingHoldingsProvider()
    first_cached = CachedHoldingsProvider(
        first,
        cache,
        "fund",
        ttl_seconds=60,
    )
    first_cached.get_holdings()

    second = _CountingHoldingsProvider(error=AssertionError("network used"))
    second_cached = CachedHoldingsProvider(
        second,
        cache,
        "fund",
        ttl_seconds=60,
    )
    result = second_cached.get_holdings()

    assert second.calls == 0
    assert second_cached.cache_status == "fresh_hit"
    assert result.iloc[0]["ticker"] == "A"
    assert second_cached.reference_fund == "ETF"


def test_cached_holdings_provider_uses_stale_frame_on_error(
    tmp_path,
    monkeypatch,
):
    cache = ProviderCache(tmp_path / "providers.sqlite3")
    monkeypatch.setattr(provider_cache.time, "time", lambda: 1_000.0)
    CachedHoldingsProvider(
        _CountingHoldingsProvider(),
        cache,
        "fund",
        ttl_seconds=60,
    ).get_holdings()

    monkeypatch.setattr(provider_cache.time, "time", lambda: 2_000.0)
    failing = _CountingHoldingsProvider(error=RuntimeError("offline"))
    cached = CachedHoldingsProvider(
        failing,
        cache,
        "fund",
        ttl_seconds=60,
    )
    result = cached.get_holdings()

    assert failing.calls == 1
    assert cached.cache_status == "stale_if_error_hit"
    assert result.iloc[0]["ticker"] == "A"
    assert any("offline" in failure for failure in cached.failures)


def test_acwi_wrapper_uses_cached_full_portfolio(tmp_path):
    class AcwiProvider:
        source_name = "acwi"
        reference_fund = "ACWI"
        holdings_as_of = "2026-07-29"
        min_calibration_rows = 1

        def __init__(self):
            self.calls = 0

        def get_holdings(self):
            self.calls += 1
            frame = pd.DataFrame(
                {
                    "ticker": ["A"],
                    "acwi_name": ["Alpha"],
                    "acwi_market_value": [100.0],
                    "acwi_weight": [1.0],
                    "acwi_location": ["United States"],
                    "acwi_exchange": ["NASDAQ"],
                }
            )
            frame.attrs["holdings_as_of"] = self.holdings_as_of
            return frame

    cache = ProviderCache(tmp_path / "providers.sqlite3")
    provider = AcwiProvider()
    cached = CachedAcwiFloatWeightsProvider(
        provider,
        cache,
        "acwi",
        ttl_seconds=60,
    )
    holdings = pd.DataFrame(
        {"ticker": ["A"], "company_name": ["Alpha"]}
    )
    market_data = pd.DataFrame(
        {
            "ticker": ["A"],
            "price": [1.0],
            "float_shares": [100.0],
            "shares_outstanding": [100.0],
            "market_cap": [100.0],
        }
    )

    first = cached.build_reference(holdings, market_data)
    second = cached.build_reference(holdings, market_data)

    assert provider.calls == 1
    assert cached.cache_status == "fresh_hit"
    assert first.iloc[0]["reference_status"] == "valid_acwi"
    assert second.iloc[0]["reference_weight_raw"] == 100.0


def test_provider_cache_key_changes_with_provider_configuration():
    first = HttpCsvHoldingsProvider(
        url="https://example.com/first.csv",
        source_name="issuer",
        reference_fund="ETF",
    )
    second = HttpCsvHoldingsProvider(
        url="https://example.com/second.csv",
        source_name="issuer",
        reference_fund="ETF",
    )

    assert provider_cache_key("fund", first) != provider_cache_key(
        "fund",
        second,
    )
