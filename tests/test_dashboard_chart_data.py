import math

import pandas as pd
import pytest

from dashboard_chart_data import prepare_constituent_weight_comparison


def test_weight_comparison_builds_excess_and_empty_gap_segments():
    components = pd.DataFrame(
        {
            "ticker": ["OVER", "UNDER", "INVALID", "SMALL"],
            "company_name": ["Over", "Under", "Invalid", "Small"],
            "actual_weight": [0.40, 0.30, 0.20, 0.10],
            "counterfactual_weight": [0.25, 0.35, 0.10, 0.10],
            "data_status": [
                "valid_acwi",
                "valid_yfinance_fallback",
                "invalid_yfinance_fallback",
                "valid_acwi",
            ],
        }
    )

    result = prepare_constituent_weight_comparison(components, limit=3)
    indexed = result.set_index("ticker")

    assert set(indexed.index) == {"OVER", "UNDER", "SMALL"}
    assert math.isclose(indexed.loc["OVER", "shared_weight"], 0.25)
    assert math.isclose(indexed.loc["OVER", "actual_excess"], 0.15)
    assert indexed.loc["OVER", "counterfactual_gap"] == 0
    assert indexed.loc["OVER", "counterfactual_label_in_shared"] == "CF 25.00%"
    assert math.isclose(indexed.loc["UNDER", "shared_weight"], 0.30)
    assert indexed.loc["UNDER", "actual_excess"] == 0
    assert math.isclose(indexed.loc["UNDER", "counterfactual_gap"], 0.05)
    assert indexed.loc["UNDER", "counterfactual_label_in_gap"] == "CF 35.00%"


def test_weight_comparison_limits_by_current_ndx_weight():
    components = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "actual_weight": [0.10, 0.30, 0.20],
            "counterfactual_weight": [0.12, 0.25, 0.18],
            "data_status": ["valid"] * 3,
        }
    )

    result = prepare_constituent_weight_comparison(components, limit=2)

    assert list(result["ticker"]) == ["C", "B"]
    assert result["actual_weight"].sum() == pytest.approx(0.50)
