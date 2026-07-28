import math

import pandas as pd

from active_share import calculate_active_share, calculate_active_share_sleeves


def test_active_share_uses_the_full_union_and_rebalanced_weights():
    ndx = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "company_name": ["Alpha", "Beta", "Gamma"],
            "actual_weight": [0.50, 0.30, 0.20],
            "rebalance_weight": [0.40, 0.35, 0.25],
        }
    )
    spx = pd.DataFrame(
        {
            "ticker": ["A", "B", "D"],
            "company_name": ["Alpha", "Beta", "Delta"],
            "actual_weight": [0.20, 0.20, 0.60],
        }
    )

    result = calculate_active_share(
        ndx,
        spx,
        spx_reference_fund="IVV",
        spx_holdings_source="test_ivv",
        spx_holdings_as_of="2026-07-24",
    )
    components = result.components.set_index("ticker")

    assert math.isclose(result.active_share, 0.60)
    assert math.isclose(result.rebalanced_active_share, 0.60)
    assert set(components.index) == {"A", "B", "C", "D"}
    assert components.loc["C", "spx_weight"] == 0
    assert components.loc["D", "ndx_weight"] == 0
    assert components.loc["D", "rebalanced_ndx_weight"] == 0


def test_active_share_normalizes_published_equity_weights_and_ticker_aliases():
    ndx = pd.DataFrame(
        {
            "ticker": ["BRK.B", "A"],
            "actual_weight": [0.2, 0.8],
        }
    )
    spx = pd.DataFrame(
        {
            "ticker": ["BRKB", "A"],
            "actual_weight": [19.0, 80.0],
        }
    )

    result = calculate_active_share(
        ndx,
        spx,
        spx_reference_fund="IVV",
        spx_holdings_source="test_ivv",
    )

    assert len(result.components) == 2
    assert result.active_share < 0.01
    assert result.rebalanced_active_share is None


def test_active_share_sleeves_are_independently_normalized():
    components = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D"],
            "company_name": ["Alpha", "Beta", "Gamma", "Delta"],
            "ndx_weight": [0.50, 0.30, 0.20, 0.00],
            "spx_weight": [0.20, 0.20, 0.00, 0.60],
            "rebalanced_ndx_weight": [0.40, 0.35, 0.25, 0.00],
        }
    )

    sleeves = calculate_active_share_sleeves(components)
    annual_sleeves = calculate_active_share_sleeves(
        components,
        ndx_weight_column="rebalanced_ndx_weight",
    )

    assert math.isclose(sleeves.ndx_active_mass, 0.60)
    assert math.isclose(sleeves.spx_active_mass, 0.60)
    assert math.isclose(sleeves.overlap_mass, 0.40)
    for column in [
        "ndx_active_weight",
        "spx_active_weight",
        "overlap_weight",
    ]:
        assert math.isclose(float(sleeves.components[column].sum()), 1.0)
        assert math.isclose(
            float(annual_sleeves.components[column].sum()),
            1.0,
        )
