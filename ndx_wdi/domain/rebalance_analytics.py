"""Analytics derived from a persisted annual Nasdaq-100 reconstitution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nasdaq100_rebalance import (
    _proportional_with_caps,
    apply_annual_security_capping,
    apply_company_capping,
)


@dataclass(frozen=True)
class AnnualRebalanceAnalysis:
    securities: pd.DataFrame
    companies: pd.DataFrame
    thresholds: pd.DataFrame
    company_redistribution: float
    security_redistribution: float
    total_capping_redistribution: float
    current_to_final_turnover: float
    company_rank_preservation_ratio: float
    security_rank_preservation_ratio: float
    beneficiary_count: int
    donor_count: int
    additions: tuple[str, ...]
    removals: tuple[str, ...]
    persisted_weight_error: float | None


def analyze_annual_rebalance(
    components: pd.DataFrame,
) -> AnnualRebalanceAnalysis:
    """Reconstruct annual weighting stages from persisted component inputs."""
    required = {
        "ticker",
        "rebalance_membership",
        "modified_market_cap_mass",
        "rebalance_company_id",
        "actual_weight",
    }
    missing = required.difference(components.columns)
    if missing:
        raise ValueError(
            f"Rebalance components are missing: {sorted(missing)}"
        )

    data = components.copy()
    data["ticker"] = data["ticker"].astype("string").str.upper().str.strip()
    for column in [
        "actual_weight",
        "modified_market_cap_mass",
        "modified_cap_ratio",
        "rebalance_weight",
    ]:
        if column not in data:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")
    membership = data["rebalance_membership"].fillna(False).astype(bool)
    selected = data.loc[
        membership & data["modified_market_cap_mass"].gt(0)
    ].copy()
    if selected.empty:
        raise ValueError(
            "No annual constituent has a positive Modified Market Cap."
        )
    selected["company_id"] = selected["rebalance_company_id"].fillna(
        selected["ticker"].map(lambda ticker: f"TICKER:{ticker}")
    )

    security_initial = selected.set_index("ticker")[
        "modified_market_cap_mass"
    ].astype(float)
    security_initial /= security_initial.sum()
    selected["initial_weight"] = selected["ticker"].map(security_initial)
    company_initial = selected.groupby("company_id")["initial_weight"].sum()

    company_stage_one = company_initial.copy()
    company_stage_one_triggered = bool(
        company_initial.gt(0.24 + 1e-12).any()
    )
    if company_stage_one_triggered:
        company_stage_one = _proportional_with_caps(
            company_initial,
            pd.Series(0.20, index=company_initial.index),
        )
    company_cohort = company_stage_one.loc[
        company_stage_one.gt(0.045 + 1e-12)
    ]
    company_final = apply_company_capping(company_initial)
    company_scale = company_final / company_initial

    company_ids = selected.set_index("ticker")["company_id"]
    security_after_company = security_initial * company_ids.map(company_scale)
    security_after_company /= security_after_company.sum()

    security_stage_one = security_after_company.copy()
    for _ in range(len(security_stage_one) + 1):
        above_trigger = security_stage_one.gt(0.15 + 1e-12)
        if not above_trigger.any():
            break
        caps = pd.Series(1.0, index=security_stage_one.index)
        caps.loc[above_trigger] = 0.14
        security_stage_one = _proportional_with_caps(
            security_stage_one,
            caps,
        )
    top_five_stage_one = security_stage_one.nlargest(
        min(5, len(security_stage_one))
    )
    security_final = apply_annual_security_capping(security_after_company)

    security = selected.set_index("ticker", drop=False).copy()
    security.index.name = None
    security["initial_weight"] = security_initial
    security["company_stage_weight"] = security_after_company
    security["security_stage_one_weight"] = security_stage_one
    security["final_weight"] = security_final
    security["capping_change"] = (
        security["final_weight"] - security["initial_weight"]
    )
    security["current_change"] = (
        security["final_weight"] - security["actual_weight"].fillna(0.0)
    )
    security["initial_rank"] = security["initial_weight"].rank(
        method="min",
        ascending=False,
    )
    security["final_rank"] = security["final_weight"].rank(
        method="min",
        ascending=False,
    )
    security["rank_change"] = (
        security["initial_rank"] - security["final_rank"]
    )

    companies = (
        selected.groupby("company_id", as_index=False)
        .agg(
            company_name=("company_name", "first"),
            tickers=("ticker", lambda values: ", ".join(sorted(values))),
        )
        .set_index("company_id")
    )
    companies["initial_weight"] = company_initial
    companies["stage_one_weight"] = company_stage_one
    companies["final_weight"] = company_final
    companies["capping_change"] = (
        companies["final_weight"] - companies["initial_weight"]
    )
    companies = companies.reset_index()

    thresholds = pd.DataFrame(
        [
            {
                "rule_id": "company_single",
                "label": "Largest company",
                "actual": float(company_initial.max()),
                "threshold": 0.24,
                "target": 0.20,
                "triggered": company_stage_one_triggered,
                "rule": "24% trigger, adjusted to no more than 20%",
            },
            {
                "rule_id": "company_cohort",
                "label": "Companies above 4.5%",
                "actual": float(company_cohort.sum()),
                "threshold": 0.48,
                "target": 0.40,
                "triggered": bool(
                    float(company_cohort.sum()) >= 0.48 - 1e-12
                ),
                "rule": "48% cohort trigger, adjusted to 40%",
            },
            {
                "rule_id": "security_single",
                "label": "Largest security",
                "actual": float(security_after_company.max()),
                "threshold": 0.15,
                "target": 0.14,
                "triggered": bool(
                    security_after_company.gt(0.15 + 1e-12).any()
                ),
                "rule": "15% trigger, adjusted to no more than 14%",
            },
            {
                "rule_id": "security_top_five",
                "label": "Five largest securities",
                "actual": float(top_five_stage_one.sum()),
                "threshold": 0.40,
                "target": 0.385,
                "triggered": bool(
                    float(top_five_stage_one.sum()) >= 0.40 - 1e-12
                ),
                "rule": "40% top-five trigger, adjusted to 38.5%",
            },
        ]
    )
    thresholds["distance_to_trigger"] = (
        thresholds["threshold"] - thresholds["actual"]
    )

    all_tickers = data.set_index("ticker")
    current_weights = all_tickers["actual_weight"].fillna(0.0).clip(lower=0)
    if float(current_weights.sum()) > 0:
        current_weights /= current_weights.sum()
    final_on_union = security_final.reindex(
        current_weights.index,
        fill_value=0.0,
    )

    company_rank_preservation = _rank_preservation_ratio(
        company_initial,
        company_final,
    )
    security_rank_preservation = _rank_preservation_ratio(
        security_after_company,
        security_final,
    )

    persisted_error: float | None = None
    persisted = security["rebalance_weight"].dropna()
    if not persisted.empty:
        persisted_error = float(
            (
                persisted
                - security_final.reindex(persisted.index)
            ).abs().max()
        )

    additions = tuple(
        sorted(
            data.loc[
                membership & data["actual_weight"].fillna(0).le(0),
                "ticker",
            ].dropna()
        )
    )
    removals = tuple(
        sorted(
            data.loc[
                ~membership & data["actual_weight"].fillna(0).gt(0),
                "ticker",
            ].dropna()
        )
    )
    company_redistribution = 0.5 * float(
        (security_after_company - security_initial).abs().sum()
    )
    security_redistribution = 0.5 * float(
        (security_final - security_after_company).abs().sum()
    )
    total_redistribution = 0.5 * float(
        (security_final - security_initial).abs().sum()
    )
    current_turnover = 0.5 * float(
        (final_on_union - current_weights).abs().sum()
    )
    return AnnualRebalanceAnalysis(
        securities=security.reset_index(drop=True),
        companies=companies,
        thresholds=thresholds,
        company_redistribution=company_redistribution,
        security_redistribution=security_redistribution,
        total_capping_redistribution=total_redistribution,
        current_to_final_turnover=current_turnover,
        company_rank_preservation_ratio=company_rank_preservation,
        security_rank_preservation_ratio=security_rank_preservation,
        beneficiary_count=int(security["capping_change"].gt(1e-12).sum()),
        donor_count=int(security["capping_change"].lt(-1e-12).sum()),
        additions=additions,
        removals=removals,
        persisted_weight_error=persisted_error,
    )


def _rank_preservation_ratio(
    before: pd.Series,
    after: pd.Series,
) -> float:
    """Return the share of strict pairwise rank relationships not inverted."""
    initial = pd.to_numeric(before, errors="coerce").dropna()
    final = pd.to_numeric(after, errors="coerce").reindex(initial.index)
    order = initial.sort_values(ascending=False).index
    final_values = final.reindex(order).to_numpy()
    inversion_count = 0
    for left in range(len(final_values)):
        inversion_count += int(
            np.count_nonzero(
                final_values[left] + 1e-12
                < final_values[left + 1 :]
            )
        )
    possible_pairs = len(final_values) * (len(final_values) - 1) / 2
    return (
        1.0 - inversion_count / possible_pairs
        if possible_pairs
        else 1.0
    )
