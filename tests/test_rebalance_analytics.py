import math

import pandas as pd

from rebalance_analytics import analyze_annual_rebalance


def test_annual_rebalance_analysis_reconstructs_stages_and_thresholds():
    tickers = [f"C{index}" for index in range(20)]
    initial = [0.26] + [0.06] * 5 + [0.44 / 14] * 14
    components = pd.DataFrame(
        {
            "ticker": tickers,
            "company_name": tickers,
            "actual_weight": initial,
            "rebalance_membership": True,
            "rebalance_company_id": tickers,
            "modified_market_cap_mass": initial,
            "modified_cap_ratio": 1.0,
            "rebalance_weight": None,
        }
    )

    analysis = analyze_annual_rebalance(components)
    thresholds = analysis.thresholds.set_index("rule_id")

    assert math.isclose(analysis.securities["initial_weight"].sum(), 1.0)
    assert math.isclose(analysis.securities["final_weight"].sum(), 1.0)
    assert thresholds.loc["company_single", "triggered"]
    assert thresholds.loc["company_cohort", "triggered"]
    assert thresholds.loc["security_single", "triggered"]
    assert analysis.company_redistribution > 0
    assert analysis.security_redistribution > 0
    assert analysis.company_rank_preservation_ratio == 1.0
    assert analysis.security_rank_preservation_ratio == 1.0
    assert analysis.persisted_weight_error is None


def test_annual_rebalance_analysis_tracks_entries_and_removals():
    tickers = [f"A{index}" for index in range(19)] + ["NEW"]
    initial = [0.08] * 4 + [0.68 / 16] * 16
    components = pd.DataFrame(
        {
            "ticker": tickers,
            "company_name": tickers,
            "actual_weight": initial[:-1] + [0.0],
            "rebalance_membership": True,
            "rebalance_company_id": tickers,
            "modified_market_cap_mass": initial,
            "modified_cap_ratio": 1.0,
            "rebalance_weight": initial,
        }
    )
    components.loc[len(components)] = {
        "ticker": "OLD",
        "company_name": "Old",
        "actual_weight": initial[-1],
        "rebalance_membership": False,
        "rebalance_company_id": "OLD",
        "modified_market_cap_mass": 0.0,
        "modified_cap_ratio": 0.0,
        "rebalance_weight": 0.0,
    }

    analysis = analyze_annual_rebalance(components)

    assert analysis.additions == ("NEW",)
    assert analysis.removals == ("OLD",)
    assert analysis.current_to_final_turnover > 0
