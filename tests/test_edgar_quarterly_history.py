import math

import pandas as pd

from edgar_quarterly_history import (
    _is_edgar_submission,
    _rebalance_type,
    calculate_quarterly_components,
    calculate_quarterly_point,
    parse_nport_positions,
)


def test_edgar_submission_detection_is_case_insensitive_and_allows_preamble():
    assert _is_edgar_submission("<edgarsubmission><header />")
    assert _is_edgar_submission(" \n<?xml version='1.0'?><EDGARSUBMISSION>")
    assert not _is_edgar_submission("<html><body>SEC rate limit</body></html>")


def test_parse_complete_nport_xml_with_bare_ampersand():
    raw = """<edgarsubmission>
    <invstorsecs>
      <invstorsec>
        <name>Alpha & Beta Inc</name>
        <title>Alpha & Beta Inc, Class A</title>
        <cusip>123456789</cusip>
        <balance>100</balance>
        <units>NS</units>
        <curcd>USD</curcd>
        <valusd>1000</valusd>
        <pctval>1.25</pctval>
        <payoffprofile>Long</payoffprofile>
        <assetcat>EC</assetcat>
      </invstorsec>
      <invstorsec>
        <name>Cash</name>
        <cusip>000000000</cusip>
        <balance>100</balance>
        <units>NS</units>
        <curcd>USD</curcd>
        <valusd>100</valusd>
        <pctval>0.1</pctval>
        <payoffprofile>Long</payoffprofile>
        <assetcat>STIV</assetcat>
      </invstorsec>
    </invstorsecs>
    </edgarsubmission>"""

    result = parse_nport_positions(raw)

    assert result["cusip"].tolist() == ["123456789"]
    assert result.loc[0, "security_name"] == "Alpha & Beta Inc, Class A"
    assert result.loc[0, "pct_value"] == 1.25


def test_parse_flattened_reader_fallback():
    raw = """Issuer ABCDEFGHIJKLMNOPQRST Alpha Inc, Class A
123456789 100 NS USD 1000 1.25 Long EC CORP US
Issuer ABCDEFGHIJKLMNOPQRSU Bond
987654321 10 NS USD 100 0.10 Long DB CORP US
"""

    result = parse_nport_positions(raw)

    assert result["cusip"].tolist() == ["123456789"]
    assert result.loc[0, "pct_value"] == 1.25


def test_exact_cusip_intersection_is_renormalized_for_both_funds():
    matched_cusips = [f"A{index:08d}" for index in range(40)]
    qqq = pd.DataFrame(
        {
            "cusip": matched_cusips + ["ADR000001"],
            "security_name": [f"Company {index}" for index in range(40)]
            + ["Depositary receipt"],
            "pct_value": [1.0] * 41,
        }
    )
    spgm = pd.DataFrame(
        {
            "cusip": matched_cusips + ["ORD000001"],
            "security_name": [f"Company {index}" for index in range(40)]
            + ["Primary listing"],
            "pct_value": list(range(1, 41)) + [10.0],
        }
    )

    components = calculate_quarterly_components(qqq, spgm)
    point = calculate_quarterly_point(qqq, spgm)
    matched = components.loc[components["match_status"].eq("matched")]
    excluded = components.loc[
        components["match_status"].eq("excluded_not_in_spgm")
    ]

    assert len(matched) == 40
    assert excluded["cusip"].tolist() == ["ADR000001"]
    assert math.isclose(matched["qqq_weight"].sum(), 1.0)
    assert math.isclose(matched["spgm_weight"].sum(), 1.0)
    assert point["matched_count"] == 40
    assert point["excluded_qqq_count"] == 1
    assert point["estimated_count"] == 0
    assert point["excluded_non_comparable_count"] == 1
    assert math.isclose(point["coverage_ratio"], 1.0)
    assert math.isclose(
        point["ndx_wdi_raw"], matched["distortion_contribution"].sum()
    )
    assert math.isclose(point["ndx_wdi"], point["ndx_wdi_raw"])


def test_median_correction_recovers_a_constant_missing_overweight_ratio():
    cusips = [f"A{index:08d}" for index in range(50)]
    qqq_weights = [2.0] * 20 + [1.0] * 25 + [2.0] * 5
    qqq = pd.DataFrame(
        {
            "cusip": cusips,
            "security_name": [f"Company {index}" for index in range(50)],
            "pct_value": qqq_weights,
        }
    )
    observed_spgm = pd.DataFrame(
        {
            "cusip": cusips[:45],
            "security_name": [f"Company {index}" for index in range(45)],
            "pct_value": [1.0] * 45,
        }
    )
    complete_spgm = pd.DataFrame(
        {
            "cusip": cusips,
            "security_name": [f"Company {index}" for index in range(50)],
            "pct_value": [1.0] * 50,
        }
    )

    corrected = calculate_quarterly_point(qqq, observed_spgm)
    complete = calculate_quarterly_point(qqq, complete_spgm)
    components = calculate_quarterly_components(qqq, observed_spgm)

    assert corrected["estimated_count"] == 5
    assert abs(corrected["ndx_wdi"] - complete["ndx_wdi"]) < abs(
        corrected["ndx_wdi_raw"] - complete["ndx_wdi"]
    )
    assert math.isclose(corrected["ndx_wdi"], complete["ndx_wdi"])
    assert math.isclose(components["corrected_qqq_weight"].sum(), 1.0)
    assert math.isclose(components["corrected_spgm_weight"].sum(), 1.0)
    assert (
        components["correction_status"].eq("estimated_missing_spgm").sum()
        == 5
    )


def test_rebalance_labels_identify_december_and_the_2023_special_event():
    assert _rebalance_type("2022-12-31") == "annual_reconstitution"
    assert _rebalance_type("2023-09-30") == "special_rebalance"
    assert _rebalance_type("2024-06-30") == "quarterly_rebalance"
