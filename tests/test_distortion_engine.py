import math
import sys
from types import SimpleNamespace

import pandas as pd

from distortion_engine import calculate_distortion, normalize_actual_weights
from market_data_provider import (
    YFinanceMarketDataProvider,
    _allocate_shared_float_shares,
    _normalize_market_data,
)


def test_expected_ndx_wdi_is_10_and_contributions_reconcile():
    holdings = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "company_name": ["A Corp", "B Corp", "C Corp"],
            "actual_weight": [0.50, 0.30, 0.20],
        }
    )
    # price × float_shares gives market-cap proportions 60 / 25 / 15.
    market_data = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "price": [1.0, 1.0, 1.0],
            "float_shares": [60.0, 25.0, 15.0],
        }
    )

    result = calculate_distortion(holdings, market_data)
    components = result.components.set_index("ticker")

    assert result.ndx_wdi == 10.0
    assert math.isclose(components["actual_weight"].sum(), 1.0)
    assert math.isclose(components["float_weight"].sum(), 1.0)
    assert math.isclose(components["distortion_contribution"].sum(), result.ndx_wdi)
    assert math.isclose(components.loc["A", "distortion_contribution"], 5.0)
    assert math.isclose(components.loc["B", "distortion_contribution"], 2.5)
    assert math.isclose(components.loc["C", "distortion_contribution"], 2.5)


def test_normalize_actual_weights():
    holdings = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "actual_weight": [50.0, 30.0, 20.0],
        }
    )
    normalized = normalize_actual_weights(holdings)
    assert math.isclose(normalized["actual_weight"].sum(), 1.0)
    assert normalized.set_index("ticker").loc["A", "actual_weight"] == 0.5


def test_missing_float_is_excluded_without_assuming_100_percent_float():
    holdings = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "actual_weight": [0.50, 0.30, 0.20],
        }
    )
    market_data = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "price": [10.0, 20.0, 30.0],
            "float_shares": [100.0, 50.0, None],
        }
    )
    result = calculate_distortion(holdings, market_data)
    components = result.components.set_index("ticker")
    valid = result.components.loc[result.components["data_status"] == "valid"]

    assert math.isclose(result.coverage_ratio, 0.8)
    assert result.missing_float_count == 1
    assert result.snapshot_status == "partial_coverage"
    assert components.loc["C", "data_status"] == "missing_float"
    assert pd.isna(components.loc["C", "float_weight"])
    assert math.isclose(valid["actual_weight"].sum(), 1.0)
    assert math.isclose(valid["float_weight"].sum(), 1.0)


def test_missing_price_and_float_statuses_are_counted_separately():
    holdings = pd.DataFrame({"ticker": ["A", "B", "C"], "actual_weight": [0.5, 0.3, 0.2]})
    market_data = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "price": [10.0, None, None],
            "float_shares": [100.0, 50.0, None],
        }
    )
    result = calculate_distortion(holdings, market_data)
    statuses = result.components.set_index("ticker")["data_status"]

    assert result.missing_price_count == 2
    assert result.missing_float_count == 1
    assert statuses["B"] == "missing_price"
    assert statuses["C"] == "missing_price_and_float"


def test_inconsistent_float_is_excluded_instead_of_distorting_the_index():
    holdings = pd.DataFrame(
        {"ticker": ["ASML", "A", "B"], "actual_weight": [0.01, 0.49, 0.50]}
    )
    market_data = pd.DataFrame(
        {
            "ticker": ["ASML", "A", "B"],
            "price": [1747.58, 10.0, 20.0],
            "float_shares": [21_331_633_667, 100.0, 50.0],
            "shares_outstanding": [384_100_000, 120.0, 60.0],
            "market_cap": [671_245_467_648, 1_200.0, 1_200.0],
        }
    )

    result = calculate_distortion(holdings, market_data)
    components = result.components.set_index("ticker")
    valid = result.components.loc[result.components["data_status"] == "valid"]

    assert result.invalid_float_count == 1
    assert result.missing_float_count == 0
    assert math.isclose(result.coverage_ratio, 0.99)
    assert components.loc["ASML", "data_status"] == "invalid_float_inconsistent"
    assert pd.isna(components.loc["ASML", "float_weight"])
    assert math.isclose(valid["float_weight"].sum(), 1.0)


def test_shared_float_is_allocated_across_matching_share_classes():
    raw = pd.DataFrame(
        {
            "ticker": ["GOOG", "GOOGL"],
            "price": [346.12, 346.77],
            "float_shares": [10_821_405_400, 10_821_405_400],
            "shares_outstanding": [5_499_638_298, 5_867_155_790],
            "market_cap": [4_223_554_551_808, 4_231_485_980_672],
            "float_shares_status": ["reported", "reported"],
        }
    )

    allocated = _allocate_shared_float_shares(_normalize_market_data(raw)).set_index("ticker")

    assert allocated.loc["GOOG", "float_shares_status"] == "allocated_shared_float"
    assert allocated.loc["GOOGL", "float_shares_status"] == "allocated_shared_float"
    assert math.isclose(allocated["float_shares"].sum(), 10_821_405_400)
    assert allocated.loc["GOOG", "float_shares"] < allocated.loc["GOOGL", "float_shares"]


def test_total_basis_uses_outstanding_shares_and_ignores_missing_float():
    holdings = pd.DataFrame(
        {"ticker": ["A", "B", "C"], "actual_weight": [0.50, 0.30, 0.20]}
    )
    market_data = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "price": [1.0, 1.0, 1.0],
            "float_shares": [60.0, None, 10_000.0],
            "shares_outstanding": [50.0, 30.0, 20.0],
            "market_cap": [50.0, 30.0, 20.0],
        }
    )

    result = calculate_distortion(holdings, market_data, weighting_basis="total")
    components = result.components.set_index("ticker")

    assert result.weighting_basis == "total"
    assert result.ndx_wdi == 0.0
    assert result.coverage_ratio == 1.0
    assert result.missing_float_count == 1
    assert result.missing_reference_shares_count == 0
    assert result.invalid_float_count == 0
    assert components["data_status"].eq("valid").all()
    assert components["float_weight"].isna().all()
    assert math.isclose(components["counterfactual_weight"].sum(), 1.0)
    assert components.loc["C", "counterfactual_weight"] == 0.20


def test_direct_acwi_weights_do_not_require_yfinance_prices_or_float_shares():
    holdings = pd.DataFrame(
        {
            "ticker": ["A", "B", "ADR"],
            "company_name": ["Alpha", "Beta", "ADR Depositary"],
            "actual_weight": [0.50, 0.30, 0.20],
        }
    )
    reference_data = pd.DataFrame(
        {
            "ticker": ["A", "B", "ADR"],
            "price": [None, 10.0, 20.0],
            "float_shares": [None, 100.0, 500.0],
            "reference_weight_raw": [70.0, 30.0, None],
            "reference_source": [
                "ishares_acwi",
                "ishares_acwi",
                "yfinance_fallback",
            ],
            "security_type": ["Ordinary share", "Ordinary share", "ADR/ADS"],
            "acwi_weight": [0.07, 0.03, None],
            "acwi_listing": [
                "United States / NASDAQ",
                "United States / NASDAQ",
                None,
            ],
            "reference_status": [
                "valid_acwi",
                "valid_acwi",
                "invalid_yfinance_fallback",
            ],
        }
    )

    result = calculate_distortion(holdings, reference_data)
    components = result.components.set_index("ticker")

    assert math.isclose(result.coverage_ratio, 0.80)
    assert result.missing_price_count == 0
    assert result.invalid_float_count == 1
    assert result.missing_reference_shares_count == 1
    assert components.loc["A", "data_status"] == "valid_acwi"
    assert components.loc["A", "counterfactual_weight"] == 0.70
    assert components.loc["ADR", "security_type"] == "ADR/ADS"
    assert components.loc["ADR", "data_status"] == "invalid_yfinance_fallback"


def test_direct_weights_count_both_missing_price_and_float_for_fallback():
    holdings = pd.DataFrame(
        {
            "ticker": ["ACWI", "MISSING"],
            "company_name": ["ACWI Match", "Missing Fallback"],
            "actual_weight": [0.90, 0.10],
        }
    )
    reference_data = pd.DataFrame(
        {
            "ticker": ["ACWI", "MISSING"],
            "price": [None, None],
            "float_shares": [None, None],
            "reference_weight_raw": [100.0, None],
            "reference_source": ["ishares_acwi", "yfinance_fallback"],
            "reference_status": ["valid_acwi", "missing_float_yfinance_fallback"],
        }
    )

    result = calculate_distortion(holdings, reference_data)

    assert result.missing_price_count == 1
    assert result.missing_float_count == 1
    assert result.missing_reference_shares_count == 1
    assert math.isclose(result.coverage_ratio, 0.90)


def test_yfinance_cache_is_redirected_to_a_writable_local_directory(tmp_path, monkeypatch):
    captured: dict[str, str] = {}
    fake_yfinance = SimpleNamespace(
        set_tz_cache_location=lambda path: captured.setdefault("path", path)
    )
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance)
    cache_dir = tmp_path / "yfinance-cache"

    YFinanceMarketDataProvider(cache_dir=cache_dir)._configure_cache()

    assert cache_dir.is_dir()
    assert captured["path"] == str(cache_dir.resolve())


def test_yfinance_worker_count_is_clamped_to_one(monkeypatch):
    provider = YFinanceMarketDataProvider(max_workers=0)
    monkeypatch.setattr(provider, "_configure_cache", lambda: None)
    monkeypatch.setattr(
        provider,
        "_fetch_one",
        lambda ticker: {
            "ticker": ticker,
            "price": 10.0,
            "float_shares": 90.0,
            "shares_outstanding": 100.0,
            "market_cap": 1_000.0,
            "float_shares_status": "reported",
            "market_data_error": None,
        },
    )

    result = provider.get_market_data(["A"])

    assert result.loc[0, "ticker"] == "A"
    assert result.loc[0, "price"] == 10.0
