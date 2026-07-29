import math
import sys
import time
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from nasdaq100_rebalance import (
    NasdaqPublicUniverseProvider,
    SelectionResult,
    _require_known_liquidity,
    apply_annual_security_capping,
    apply_company_capping,
    derive_acwi_total_cap_conversion,
    select_annual_companies,
    select_annual_universe,
    select_quarterly_companies,
    simulate_rebalance,
)


def test_company_capping_applies_both_2026_concentration_stages():
    initial = pd.Series(
        [0.26] + [0.06] * 5 + [0.44 / 14] * 14,
        index=[f"C{i}" for i in range(20)],
    )

    result = apply_company_capping(initial)
    large = result.loc[result > 0.045]
    triggered_cohort = initial.index[initial > 0.045]

    assert math.isclose(result.sum(), 1.0)
    assert result.max() <= 0.20 + 1e-10
    assert large.sum() <= 0.40 + 1e-10
    assert math.isclose(result.loc[triggered_cohort].sum(), 0.40)
    assert list(result.sort_values(ascending=False).index) == list(
        initial.sort_values(ascending=False).index
    )


def test_annual_security_capping_enforces_top_five_and_outside_caps():
    initial = pd.Series(
        [0.16, 0.08, 0.07, 0.06, 0.05] + [0.57 / 15] * 15,
        index=[f"S{i}" for i in range(20)],
    )

    result = apply_annual_security_capping(initial)
    top_five = result.nlargest(5)

    assert math.isclose(result.sum(), 1.0)
    assert result.max() <= 0.14 + 1e-10
    assert top_five.sum() <= 0.385 + 1e-10
    assert result.drop(index=top_five.index).max() <= 0.044 + 1e-10


def test_quarterly_selection_replaces_rank_126_and_adds_fast_entry():
    companies = []
    for rank in range(1, 128):
        ticker = f"C{rank}"
        companies.append(
            {
                "ticker": ticker,
                "company_name": ticker,
                "company_id": ticker,
                "security_type": "Ordinary share",
                "company_full_market_cap": 1_000 - rank,
                "base_eligible": True,
                "regular_eligible": True,
                "fast_entry_eligible": True,
            }
        )
    companies.append(
        {
            "ticker": "IPO",
            "company_name": "Fast IPO",
            "company_id": "IPO",
            "security_type": "Ordinary share",
            "company_full_market_cap": 2_000,
            "base_eligible": True,
            "regular_eligible": False,
            "fast_entry_eligible": True,
        }
    )
    universe = pd.DataFrame(companies)
    current = [f"C{rank}" for rank in range(1, 100)] + ["C126"]

    result = select_quarterly_companies(universe, current)

    assert "C126" in result.removals
    assert "C100" in result.additions
    assert "IPO" in result.additions
    assert "IPO" in result.selected_tickers


def test_annual_selection_follows_top_75_then_retention_sequence():
    ranked = pd.DataFrame(
        {
            "company_id": [f"C{rank}" for rank in range(1, 131)],
            "full_market_cap": list(range(130, 0, -1)),
        }
    )
    previous = [f"C{rank}" for rank in range(1, 76)] + [
        f"C{rank}" for rank in range(101, 126)
    ]

    selected = select_annual_companies(
        ranked,
        previous_constituents=previous,
        previous_top_100=["C101", "C102"],
    )

    assert len(selected) == 100
    assert "C101" in selected
    assert "C102" in selected
    assert "C103" not in selected
    assert "C98" in selected
    assert "C100" not in selected


def test_annual_universe_returns_exactly_100_companies():
    universe = pd.DataFrame(
        {
            "ticker": [f"C{rank}" for rank in range(1, 131)],
            "company_name": [f"C{rank}" for rank in range(1, 131)],
            "company_id": [f"C{rank}" for rank in range(1, 131)],
            "security_type": ["Ordinary share"] * 130,
            "company_full_market_cap": list(range(130, 0, -1)),
            "base_eligible": [True] * 130,
            "regular_eligible": [True] * 130,
        }
    )
    universe = pd.concat(
        [
            universe,
            pd.DataFrame(
                {
                    "ticker": ["C1B"],
                    "company_name": ["C1 Class B"],
                    "company_id": ["C1"],
                    "security_type": ["Ordinary share"],
                    "company_full_market_cap": [130],
                    "base_eligible": [True],
                    "regular_eligible": [True],
                }
            ),
        ],
        ignore_index=True,
    )
    current = [f"C{rank}" for rank in range(1, 101)] + ["C1B"]

    result = select_annual_universe(universe, current)

    assert len(result.selected_company_ids) == 100
    assert len(result.selected_tickers) == 101
    assert {"C1", "C1B"}.issubset(result.selected_tickers)
    assert not result.additions
    assert not result.removals


def test_unknown_liquidity_aborts_selection_instead_of_excluding_security():
    universe = pd.DataFrame(
        {
            "ticker": ["KNOWN", "UNKNOWN"],
            "advt_3m": [10_000_000.0, None],
            "liquidity_status": ["available", "unknown"],
        }
    )

    with pytest.raises(ValueError, match="UNKNOWN"):
        _require_known_liquidity(universe, ["KNOWN", "UNKNOWN"])


def test_liquidity_download_has_a_global_deadline(monkeypatch):
    def slow_download(**_kwargs):
        time.sleep(0.1)
        return pd.DataFrame()

    monkeypatch.setitem(
        sys.modules,
        "yfinance",
        SimpleNamespace(download=slow_download),
    )
    provider = NasdaqPublicUniverseProvider(timeout=0.01)

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="deadline exceeded"):
        provider._download_liquidity(["SLOW"], as_of=date(2026, 7, 29))
    elapsed = time.monotonic() - started

    assert elapsed < 0.08


def test_rebalance_uses_modified_cap_ratio_and_calculates_new_wdi():
    tickers = ["A"] + [f"S{i}" for i in range(25)]
    holdings = pd.DataFrame(
        {
            "ticker": tickers,
            "company_name": tickers,
            "actual_weight": [0.50] + [0.50 / 25] * 25,
        }
    )
    securities = pd.DataFrame(
        {
            "ticker": tickers,
            "company_name": tickers,
            "company_id": tickers,
            "security_type": ["Ordinary share"] * 26,
            "is_current": [True] * 26,
            "selected": [True] * 26,
        }
    )
    selection = SelectionResult(
        securities=securities,
        selected_tickers=tuple(tickers),
        selected_company_ids=tuple(tickers),
        additions=(),
        removals=(),
        status="test",
        source="test",
        as_of="2026-07-27",
        eligible_company_count=26,
    )
    reference = pd.DataFrame(
        {
            "ticker": tickers,
            "reference_weight_raw": [30.0] + [70.0 / 25] * 25,
            "float_shares": [30.0] + [70.0 / 25] * 25,
            "shares_outstanding": [90.0] + [70.0 / 25] * 25,
        }
    )

    result = simulate_rebalance(
        holdings,
        reference,
        selection,
        rebalance_type="annual",
    )
    components = result.components.set_index("ticker")

    assert math.isclose(components["rebalance_weight"].sum(), 1.0)
    assert components.loc["A", "modified_cap_ratio"] == 3.0
    assert components.loc["A", "rebalance_weight"] == 0.14
    assert result.method == "annual_modified_market_cap_2026"
    assert result.ndx_wdi > 0


def test_asml_override_keeps_annual_modified_cap_ratio_at_one():
    tickers = ["ASML"] + [f"S{i}" for i in range(25)]
    holdings = pd.DataFrame(
        {
            "ticker": tickers,
            "company_name": tickers,
            "actual_weight": [1.0 / len(tickers)] * len(tickers),
        }
    )
    securities = pd.DataFrame(
        {
            "ticker": tickers,
            "company_name": tickers,
            "company_id": tickers,
            "security_type": ["ADR/ADS"] + ["Ordinary share"] * 25,
            "is_current": [True] * len(tickers),
            "selected": [True] * len(tickers),
        }
    )
    selection = SelectionResult(
        securities=securities,
        selected_tickers=tuple(tickers),
        selected_company_ids=tuple(tickers),
        additions=(),
        removals=(),
        status="test",
        source="test",
        as_of="2026-07-29",
        eligible_company_count=len(tickers),
    )
    reference = pd.DataFrame(
        {
            "ticker": tickers,
            "reference_source": ["hardcoded_float_override"] + [None] * 25,
            "reference_weight_raw": [88.0] + [1.0] * 25,
            "float_shares": [88_000_000.0] + [1.0] * 25,
            "shares_outstanding": [88_000_000.0] + [1.0] * 25,
            "price": [1.0] * len(tickers),
        }
    )

    result = simulate_rebalance(
        holdings,
        reference,
        selection,
        rebalance_type="annual",
    )
    asml = result.components.set_index("ticker").loc["ASML"]

    assert asml["modified_cap_ratio"] == 1.0
    assert asml["rebalance_input_status"] == "valid_hardcoded_float_override"


def test_quarterly_rebalance_does_not_recap_when_current_constraints_pass():
    tickers = ["A", "B"] + [f"S{i}" for i in range(24)]
    holdings = pd.DataFrame(
        {
            "ticker": tickers,
            "company_name": tickers,
            "actual_weight": [0.08, 0.08] + [0.84 / 24] * 24,
        }
    )
    securities = pd.DataFrame(
        {
            "ticker": tickers,
            "company_name": tickers,
            "company_id": tickers,
            "security_type": ["Ordinary share"] * len(tickers),
            "is_current": [True] * len(tickers),
            "selected": [True] * len(tickers),
        }
    )
    selection = SelectionResult(
        securities=securities,
        selected_tickers=tuple(tickers),
        selected_company_ids=tuple(tickers),
        additions=(),
        removals=(),
        status="test",
        source="test",
        as_of="2026-07-27",
        eligible_company_count=len(tickers),
    )
    reference = pd.DataFrame(
        {
            "ticker": tickers,
            # A would exceed 24% if quarterly weights were incorrectly reset
            # from raw modified capitalization.
            "reference_weight_raw": [50.0, 10.0] + [40.0 / 24] * 24,
            "float_shares": [50.0, 10.0] + [40.0 / 24] * 24,
            "shares_outstanding": [50.0, 10.0] + [40.0 / 24] * 24,
        }
    )

    result = simulate_rebalance(holdings, reference, selection)
    components = result.components.set_index("ticker")

    assert math.isclose(components.loc["A", "rebalance_weight"], 0.08)
    assert math.isclose(components.loc["B", "rebalance_weight"], 0.08)
    assert result.method == "quarterly_index_shares_then_modified_cap_2026"
    assert any(
        "no 40% cohort redistribution" in note for note in result.notes
    )


def test_direct_acwi_conversion_ignores_bad_yfinance_float_shares():
    tickers = (
        [f"FULL{i}" for i in range(20)]
        + ["LOW"]
        + [f"SMALL{i}" for i in range(80)]
    )
    holdings = pd.DataFrame(
        {
            "ticker": tickers,
            "company_name": tickers,
            "actual_weight": [1.0 / len(tickers)] * len(tickers),
        }
    )
    securities = pd.DataFrame(
        {
            "ticker": tickers,
            "company_name": tickers,
            "company_id": tickers,
            "security_type": ["Ordinary share"] * len(tickers),
            "is_current": [True] * len(tickers),
            "selected": [True] * len(tickers),
        }
    )
    selection = SelectionResult(
        securities=securities,
        selected_tickers=tuple(tickers),
        selected_company_ids=tuple(tickers),
        additions=(),
        removals=(),
        status="test",
        source="test",
        as_of="2026-07-27",
        eligible_company_count=len(tickers),
    )
    reference = pd.DataFrame(
        {
            "ticker": tickers,
            "reference_source": ["ishares_acwi"] * len(tickers),
            "reference_weight_raw": [10.0] * 20 + [1.0] * 81,
            "modified_float_mass_raw": [10.0] * 20 + [1.0] * 81,
            "counterfactual_reference_raw": [10.0] * 20 + [1.0] * 81,
            "price": [1.0] * len(tickers),
            "shares_outstanding": [1_000.0] * 21 + [100.0] * 80,
            # These values are deliberately impossible and must not affect
            # direct-ACWI modified capitalization.
            "float_shares": [1.0] * 20 + [1_000_000.0] + [1.0] * 80,
        }
    )

    conversion_data = reference.assign(
        listed_total_cap=reference["price"] * reference["shares_outstanding"]
    )
    scale, count = derive_acwi_total_cap_conversion(conversion_data)
    result = simulate_rebalance(holdings, reference, selection)
    components = result.components.set_index("ticker")

    assert math.isclose(scale or 0.0, 0.01)
    assert count == 101
    assert math.isclose(result.acwi_conversion_scale or 0.0, 0.01)
    assert result.acwi_calibration_count == 101
    assert components.loc["FULL0", "modified_cap_ratio"] == 1.0
    assert components.loc["LOW", "modified_cap_ratio"] == 3.0
    assert set(components["rebalance_input_status"]) == {
        "valid_acwi_converted_tso"
    }
