"""Pure calculation engine for the Nasdaq-100 Weight Distortion Index."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


COMPONENT_COLUMNS = [
    "ticker",
    "company_name",
    "actual_weight",
    "float_weight",
    "counterfactual_weight",
    "weight_delta",
    "weight_ratio",
    "distortion_contribution",
    "price",
    "float_shares",
    "reference_shares",
    "security_type",
    "reference_source",
    "acwi_weight",
    "acwi_listing",
    "data_status",
]


@dataclass(frozen=True)
class DistortionResult:
    components: pd.DataFrame
    ndx_wdi: float
    coverage_ratio: float
    constituent_count: int
    missing_float_count: int
    invalid_float_count: int
    missing_reference_shares_count: int
    missing_price_count: int
    snapshot_status: str
    weighting_basis: str


def normalize_actual_weights(holdings: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize positive holdings weights to exactly one."""
    required = {"ticker", "actual_weight"}
    missing = required.difference(holdings.columns)
    if missing:
        raise ValueError(f"Missing holdings columns: {sorted(missing)}")
    result = holdings.copy()
    result["actual_weight"] = pd.to_numeric(result["actual_weight"], errors="coerce")
    result = result.loc[result["actual_weight"].notna() & (result["actual_weight"] > 0)].copy()
    if result.empty:
        raise ValueError("No positive published weights are available.")
    result["ticker"] = result["ticker"].astype("string").str.upper().str.strip()
    if "company_name" not in result:
        result["company_name"] = result["ticker"]
    result = (
        result.groupby("ticker", as_index=False, sort=False)
        .agg(company_name=("company_name", "first"), actual_weight=("actual_weight", "sum"))
    )
    result["actual_weight"] = result["actual_weight"] / result["actual_weight"].sum()
    return result


def calculate_distortion(
    holdings: pd.DataFrame,
    market_data: pd.DataFrame,
    *,
    coverage_threshold: float = 0.99,
    weighting_basis: str = "float",
    float_shares_tolerance: float = 1.10,
    float_cap_tolerance: float = 1.25,
) -> DistortionResult:
    """Compare published weights with an uncapped capitalization counterfactual.

    Invalid rows are retained for transparency but excluded from the score. The
    valid actual weights are re-normalized to the covered universe, while
    coverage_ratio reports the original QQQ weight represented by that universe.
    """
    if weighting_basis not in {"float", "total"}:
        raise ValueError("weighting_basis must be 'float' or 'total'.")
    normalized = normalize_actual_weights(holdings)
    data = normalized.merge(market_data, on="ticker", how="left")
    for column in ["price", "float_shares", "shares_outstanding", "market_cap"]:
        if column not in data:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column, default in [
        ("security_type", "Ordinary share"),
        ("reference_source", None),
        ("acwi_weight", np.nan),
        ("acwi_listing", None),
    ]:
        if column not in data:
            data[column] = default

    missing_price = data["price"].isna() | (data["price"] <= 0)
    missing_float = data["float_shares"].isna() | (data["float_shares"] <= 0)
    has_outstanding = data["shares_outstanding"].notna() & (data["shares_outstanding"] > 0)
    has_market_cap = data["market_cap"].notna() & (data["market_cap"] > 0)
    implied_float_cap = data["price"] * data["float_shares"]
    float_gt_outstanding = (
        ~missing_float
        & has_outstanding
        & (data["float_shares"] > data["shares_outstanding"] * float_shares_tolerance)
    )
    float_cap_gt_market_cap = (
        ~missing_price
        & ~missing_float
        & has_market_cap
        & (implied_float_cap > data["market_cap"] * float_cap_tolerance)
    )
    invalid_float = float_gt_outstanding | float_cap_gt_market_cap
    missing_outstanding = data["shares_outstanding"].isna() | (
        data["shares_outstanding"] <= 0
    )
    uses_direct_float_weights = (
        weighting_basis == "float" and "reference_weight_raw" in data
    )
    if uses_direct_float_weights:
        reference_weight_raw = pd.to_numeric(
            data["reference_weight_raw"], errors="coerce"
        )
        missing_reference_shares = (
            reference_weight_raw.isna() | (reference_weight_raw <= 0)
        )
        reference_status = data.get(
            "reference_status",
            pd.Series("missing_float_reference", index=data.index),
        ).fillna("missing_float_reference")
        invalid_reference_shares = reference_status.eq(
            "invalid_yfinance_fallback"
        )
        valid = ~missing_reference_shares
        reference_shares = pd.Series(np.nan, index=data.index)
        missing_float_for_result = reference_status.eq(
            "missing_float_yfinance_fallback"
        )
        missing_price_for_result = reference_status.eq(
            "missing_price_yfinance_fallback"
        )
        invalid_float_for_result = invalid_reference_shares
    elif weighting_basis == "float":
        reference_shares = data["float_shares"]
        missing_reference_shares = missing_float
        invalid_reference_shares = invalid_float
        valid = ~(missing_price | missing_reference_shares | invalid_reference_shares)
        missing_float_for_result = missing_float
        missing_price_for_result = missing_price
        invalid_float_for_result = invalid_float
    elif weighting_basis == "total":
        reference_shares = data["shares_outstanding"]
        missing_reference_shares = missing_outstanding
        invalid_reference_shares = pd.Series(False, index=data.index)
        valid = ~(missing_price | missing_reference_shares)
        missing_float_for_result = missing_float
        missing_price_for_result = missing_price
        invalid_float_for_result = pd.Series(False, index=data.index)

    if uses_direct_float_weights:
        data["data_status"] = reference_status
    else:
        data["data_status"] = "valid"
    if (
        weighting_basis == "float"
        and not uses_direct_float_weights
        and "float_shares_status" in data
    ):
        allocated_float = data["float_shares_status"].eq("allocated_shared_float")
        data.loc[valid & allocated_float, "data_status"] = "valid_shared_float_allocated"
    if weighting_basis == "float" and not uses_direct_float_weights:
        data.loc[missing_price & ~missing_float, "data_status"] = "missing_price"
        data.loc[missing_float & ~missing_price, "data_status"] = "missing_float"
        data.loc[missing_price & missing_float, "data_status"] = "missing_price_and_float"
        data.loc[invalid_float, "data_status"] = "invalid_float_inconsistent"
    elif weighting_basis == "total":
        data.loc[missing_price & ~missing_outstanding, "data_status"] = "missing_price"
        data.loc[
            missing_outstanding & ~missing_price, "data_status"
        ] = "missing_shares_outstanding"
        data.loc[
            missing_price & missing_outstanding, "data_status"
        ] = "missing_price_and_shares_outstanding"

    coverage_ratio = float(data.loc[valid, "actual_weight"].sum())
    if not valid.any():
        source_errors: list[str] = []
        if "market_data_error" in data:
            source_errors = [
                str(error)
                for error in data["market_data_error"].dropna().unique().tolist()
                if str(error).strip()
            ]
        detail = f" Provider error: {source_errors[0]}" if source_errors else ""
        requirement = (
            "a valid ACWI or calibrated fallback weight"
            if uses_direct_float_weights
            else "both a valid price and valid reference shares"
        )
        raise ValueError(f"No security has {requirement}." + detail)

    data["float_weight"] = np.nan
    data["counterfactual_weight"] = np.nan
    data["weight_delta"] = np.nan
    data["weight_ratio"] = np.nan
    data["distortion_contribution"] = np.nan

    covered_actual = data.loc[valid, "actual_weight"] / coverage_ratio
    if uses_direct_float_weights:
        reference_market_cap = reference_weight_raw.loc[valid]
    else:
        reference_market_cap = data.loc[valid, "price"] * reference_shares.loc[valid]
    reference_cap_total = float(reference_market_cap.sum())
    if not np.isfinite(reference_cap_total) or reference_cap_total <= 0:
        raise ValueError("Total reference market capitalization must be positive.")
    counterfactual_weight = reference_market_cap / reference_cap_total
    delta = covered_actual - counterfactual_weight

    data.loc[valid, "actual_weight"] = covered_actual
    if weighting_basis == "float":
        data.loc[valid, "float_weight"] = counterfactual_weight
    data.loc[valid, "counterfactual_weight"] = counterfactual_weight
    data.loc[valid, "weight_delta"] = delta
    data.loc[valid, "weight_ratio"] = covered_actual / counterfactual_weight
    data.loc[valid, "distortion_contribution"] = 50.0 * delta.abs()
    data["reference_shares"] = reference_shares

    ndx_wdi = float(data.loc[valid, "distortion_contribution"].sum())
    if coverage_ratio >= coverage_threshold:
        snapshot_status = "complete"
    else:
        snapshot_status = "partial_coverage"

    data = data[COMPONENT_COLUMNS].sort_values(
        ["distortion_contribution", "ticker"], ascending=[False, True], na_position="last"
    )
    return DistortionResult(
        components=data.reset_index(drop=True),
        ndx_wdi=ndx_wdi,
        coverage_ratio=coverage_ratio,
        constituent_count=len(data),
        missing_float_count=int(missing_float_for_result.sum()),
        invalid_float_count=int(invalid_float_for_result.sum()),
        missing_reference_shares_count=int(missing_reference_shares.sum()),
        missing_price_count=int(missing_price_for_result.sum()),
        snapshot_status=snapshot_status,
        weighting_basis=weighting_basis,
    )
