import math

import pandas as pd

from acwi_weights_provider import (
    add_yfinance_fallbacks,
    match_acwi_holdings,
    parse_acwi_holdings_csv,
)


def test_acwi_parser_preserves_unrounded_market_values():
    raw = """iShares MSCI ACWI ETF
Fund Holdings as of,"Jul 24, 2026"
Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Location,Exchange
A,ALPHA INC,Technology,Equity,"1,234,567.89",0.01,United States,NASDAQ
USD,US DOLLAR,Cash and/or Derivatives,Cash,"10,000",0.00,United States,-
"""

    result = parse_acwi_holdings_csv(raw)

    assert result["ticker"].tolist() == ["A"]
    assert result.loc[0, "acwi_market_value"] == 1_234_567.89
    assert result.loc[0, "acwi_weight"] == 0.0001
    assert result.attrs["holdings_as_of"] == "Jul 24, 2026"


def test_name_matching_resolves_ticker_collisions_and_excludes_adrs():
    holdings = pd.DataFrame(
        {
            "ticker": ["ADP", "ROP", "ASML"],
            "company_name": [
                "AUTOMATIC DATA PROCESSING INC",
                "ROPER TECHNOLOGIES INC",
                "ASML HOLDING ADR REPRESENTING NV",
            ],
        }
    )
    acwi = pd.DataFrame(
        {
            "ticker": ["ADP", "ADP", "ROP", "ROP", "ASML"],
            "acwi_name": [
                "AEROPORTS DE PARIS SA",
                "AUTOMATIC DATA PROCESSING INC",
                "ROCHE PS PAR AG",
                "ROPER TECHNOLOGIES INC",
                "ASML HOLDING NV",
            ],
            "acwi_market_value": [1.0, 100.0, 300.0, 40.0, 700.0],
            "acwi_weight": [0.0, 0.001, 0.003, 0.0004, 0.007],
            "acwi_location": [
                "France",
                "United States",
                "Switzerland",
                "United States",
                "Netherlands",
            ],
            "acwi_exchange": [
                "Euronext Paris",
                "NASDAQ",
                "SIX Swiss Exchange",
                "NASDAQ",
                "Euronext Amsterdam",
            ],
        }
    )

    result = match_acwi_holdings(holdings, acwi).set_index("ticker")

    assert result.loc["ADP", "reference_weight_raw"] == 100.0
    assert result.loc["ROP", "reference_weight_raw"] == 40.0
    assert result.loc["ASML", "security_type"] == "ADR/ADS"
    assert result.loc["ASML", "reference_source"] == "yfinance_fallback"
    assert pd.isna(result.loc["ASML", "reference_weight_raw"])


def test_yfinance_fallback_is_calibrated_to_acwi_market_value_units():
    references = pd.DataFrame(
        {
            "ticker": ["A", "B", "ADR"],
            "reference_weight_raw": [100.0, 200.0, None],
            "reference_source": [
                "ishares_acwi",
                "ishares_acwi",
                "yfinance_fallback",
            ],
            "security_type": ["Ordinary share", "Ordinary share", "ADR/ADS"],
            "acwi_weight": [0.01, 0.02, None],
            "acwi_market_value": [100.0, 200.0, None],
            "acwi_listing": ["US / NASDAQ", "US / NASDAQ", None],
            "reference_status": [
                "valid_acwi",
                "valid_acwi",
                "pending_yfinance_fallback",
            ],
        }
    )
    market_data = pd.DataFrame(
        {
            "ticker": ["A", "B", "ADR"],
            "price": [10.0, 10.0, 10.0],
            "float_shares": [10.0, 20.0, 5.0],
            "shares_outstanding": [12.0, 22.0, 6.0],
            "market_cap": [120.0, 220.0, 60.0],
        }
    )

    result = add_yfinance_fallbacks(
        references, market_data, min_calibration_rows=2
    ).set_index("ticker")

    assert math.isclose(result.loc["ADR", "reference_weight_raw"], 50.0)
    assert result.loc["ADR", "reference_status"] == "valid_yfinance_fallback"


def test_asml_override_is_not_reported_as_yfinance_fallback():
    references = pd.DataFrame(
        {
            "ticker": ["A", "B", "ASML"],
            "reference_weight_raw": [100.0, 200.0, None],
            "reference_source": [
                "ishares_acwi",
                "ishares_acwi",
                "yfinance_fallback",
            ],
            "security_type": [
                "Ordinary share",
                "Ordinary share",
                "ADR/ADS",
            ],
            "acwi_weight": [0.01, 0.02, None],
            "acwi_market_value": [100.0, 200.0, None],
            "acwi_listing": ["US / NASDAQ", "US / NASDAQ", None],
            "reference_status": [
                "valid_acwi",
                "valid_acwi",
                "pending_yfinance_fallback",
            ],
        }
    )
    market_data = pd.DataFrame(
        {
            "ticker": ["A", "B", "ASML"],
            "price": [10.0, 10.0, 10.0],
            "float_shares": [10.0, 20.0, 88_000_000.0],
            "shares_outstanding": [12.0, 22.0, 384_100_000.0],
            "market_cap": [120.0, 220.0, 3_841_000_000.0],
            "float_shares_status": [
                "reported",
                "reported",
                "hardcoded_float_override",
            ],
        }
    )

    result = add_yfinance_fallbacks(
        references, market_data, min_calibration_rows=2
    ).set_index("ticker")

    assert result.loc["ASML", "reference_weight_raw"] == 880_000_000.0
    assert (
        result.loc["ASML", "reference_source"]
        == "hardcoded_float_override"
    )
    assert (
        result.loc["ASML", "reference_status"]
        == "valid_hardcoded_float_override"
    )
