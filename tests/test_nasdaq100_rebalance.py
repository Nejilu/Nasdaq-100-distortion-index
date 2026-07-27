import math

import pandas as pd

from nasdaq100_rebalance import (
    SelectionResult,
    apply_annual_security_capping,
    apply_company_capping,
    select_annual_companies,
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

    assert math.isclose(result.sum(), 1.0)
    assert result.max() <= 0.20 + 1e-10
    assert large.sum() <= 0.40 + 1e-10
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

    result = simulate_rebalance(holdings, reference, selection)
    components = result.components.set_index("ticker")

    assert math.isclose(components["rebalance_weight"].sum(), 1.0)
    assert components.loc["A", "modified_cap_ratio"] == 3.0
    assert components.loc["A", "rebalance_weight"] == 0.20
    assert result.ndx_wdi > 0
