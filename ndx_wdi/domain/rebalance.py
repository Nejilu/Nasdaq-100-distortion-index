"""Pure Nasdaq-100 selection and weighting engines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SelectionResult:
    securities: pd.DataFrame
    selected_tickers: tuple[str, ...]
    selected_company_ids: tuple[str, ...]
    additions: tuple[str, ...]
    removals: tuple[str, ...]
    status: str
    source: str
    as_of: str
    eligible_company_count: int
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RebalanceResult:
    components: pd.DataFrame
    ndx_wdi: float
    coverage_ratio: float
    constituent_count: int
    status: str
    method: str
    reference_date: str
    additions: tuple[str, ...]
    removals: tuple[str, ...]
    data_source: str
    acwi_conversion_scale: float | None = None
    acwi_calibration_count: int = 0
    notes: tuple[str, ...] = ()


def fallback_current_selection(
    holdings: pd.DataFrame,
    *,
    as_of: date | None = None,
    reason: str,
) -> SelectionResult:
    """Preserve current membership when the public universe cannot be loaded."""
    as_of = as_of or date.today()
    securities = holdings[["ticker", "company_name"]].copy()
    securities["ticker"] = securities["ticker"].astype("string").str.upper().str.strip()
    securities["company_id"] = securities["ticker"].map(lambda value: f"TICKER:{value}")
    securities["security_type"] = "Ordinary share"
    securities["is_current"] = True
    securities["selected"] = True
    tickers = tuple(securities["ticker"])
    company_ids = tuple(securities["company_id"])
    return SelectionResult(
        securities=securities,
        selected_tickers=tickers,
        selected_company_ids=company_ids,
        additions=(),
        removals=(),
        status="current_composition_fallback",
        source="current_holdings_only",
        as_of=as_of.isoformat(),
        eligible_company_count=len(company_ids),
        notes=(reason,),
    )


def select_quarterly_companies(
    universe: pd.DataFrame,
    current_tickers: Iterable[str],
) -> SelectionResult:
    """Apply the current quarterly rank, replacement and fast-entry rules."""
    data = universe.copy()
    current_tickers = {str(value).upper().strip() for value in current_tickers}
    data["ticker"] = data["ticker"].astype("string").str.upper().str.strip()
    data["is_current"] = data["ticker"].isin(current_tickers)
    for column, default in [
        ("regular_eligible", False),
        ("fast_entry_eligible", False),
        ("company_full_market_cap", np.nan),
    ]:
        if column not in data:
            data[column] = default

    company = (
        data.sort_values("company_full_market_cap", ascending=False, na_position="last")
        .groupby("company_id", as_index=False, sort=False)
        .agg(
            company_full_market_cap=("company_full_market_cap", "max"),
            is_current=("is_current", "max"),
            regular_eligible=("regular_eligible", "max"),
            fast_entry_eligible=("fast_entry_eligible", "max"),
        )
        .sort_values("company_full_market_cap", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    ranked = company.loc[company["regular_eligible"] | company["is_current"]].copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    current_companies = set(ranked.loc[ranked["is_current"], "company_id"])
    selected = set(current_companies)

    removals = ranked.loc[
        ranked["is_current"] & ranked["rank"].gt(125), "company_id"
    ].tolist()
    for company_id in reversed(removals):
        selected.discard(company_id)

    target_count = len(current_companies)
    replacements = ranked.loc[
        ranked["regular_eligible"] & ~ranked["company_id"].isin(selected)
    ]
    for company_id in replacements["company_id"]:
        if len(selected) >= target_count:
            break
        selected.add(company_id)

    current_ranked = ranked.loc[ranked["company_id"].isin(current_companies)]
    top_40_threshold = (
        float(current_ranked.iloc[min(39, len(current_ranked) - 1)]["company_full_market_cap"])
        if not current_ranked.empty
        else np.inf
    )
    fast_entries = company.loc[
        company["fast_entry_eligible"]
        & ~company["company_id"].isin(selected)
        & company["company_full_market_cap"].ge(top_40_threshold)
    ]
    selected.update(fast_entries["company_id"])

    selected_rows = data.loc[data["company_id"].isin(selected)].copy()
    # Existing companies retain their currently represented securities. New
    # companies bring all otherwise eligible listed classes into the simulation.
    selected_rows["selected"] = selected_rows["is_current"] | selected_rows.get(
        "base_eligible", True
    )
    selected_rows = selected_rows.loc[selected_rows["selected"]].copy()
    selected_tickers = tuple(selected_rows["ticker"].drop_duplicates())
    selected_current = set(selected_tickers).intersection(current_tickers)
    additions = tuple(sorted(set(selected_tickers).difference(current_tickers)))
    removals_tickers = tuple(sorted(current_tickers.difference(selected_current)))

    data["selected"] = data["ticker"].isin(selected_tickers)
    securities = data.loc[
        data["is_current"] | data["selected"],
        [
            "ticker",
            "company_name",
            "company_id",
            "security_type",
            "is_current",
            "selected",
        ],
    ].drop_duplicates("ticker", keep="first")
    return SelectionResult(
        securities=securities,
        selected_tickers=selected_tickers,
        selected_company_ids=tuple(sorted(selected)),
        additions=additions,
        removals=removals_tickers,
        status="calculated",
        source="provided_universe",
        as_of=date.today().isoformat(),
        eligible_company_count=int(ranked["regular_eligible"].sum()),
        notes=(),
    )


def select_annual_companies(
    ranked_companies: pd.DataFrame,
    *,
    previous_constituents: Iterable[str],
    previous_top_100: Iterable[str],
    added_since_reconstitution: Iterable[str] = (),
) -> tuple[str, ...]:
    """Apply the annual top-75/100/125 company selection sequence."""
    ranked = ranked_companies.sort_values("full_market_cap", ascending=False).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    previous = set(previous_constituents)
    protected = set(previous_top_100).union(added_since_reconstitution)
    selected: list[str] = []

    def add(values: Iterable[str]) -> None:
        for value in values:
            if value not in selected and len(selected) < 100:
                selected.append(value)

    add(ranked.loc[ranked["rank"].le(75), "company_id"])
    add(ranked.loc[ranked["company_id"].isin(previous) & ranked["rank"].le(100), "company_id"])
    add(
        ranked.loc[
            ranked["company_id"].isin(previous.intersection(protected))
            & ranked["rank"].between(101, 125),
            "company_id",
        ]
    )
    add(ranked.loc[ranked["rank"].le(100), "company_id"])
    return tuple(selected)


def select_annual_universe(
    universe: pd.DataFrame,
    current_tickers: Iterable[str],
) -> SelectionResult:
    """Apply annual eligibility and the top-75/100/125 selection sequence."""
    data = universe.copy()
    current_tickers = {str(value).upper().strip() for value in current_tickers}
    data["ticker"] = data["ticker"].astype("string").str.upper().str.strip()
    data["is_current"] = data["ticker"].isin(current_tickers)
    for column, default in [
        ("regular_eligible", False),
        ("company_full_market_cap", np.nan),
    ]:
        if column not in data:
            data[column] = default

    company = (
        data.loc[data["regular_eligible"]]
        .sort_values("company_full_market_cap", ascending=False, na_position="last")
        .groupby("company_id", as_index=False, sort=False)
        .agg(full_market_cap=("company_full_market_cap", "max"))
        .sort_values("full_market_cap", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    current_companies = set(
        data.loc[data["is_current"], "company_id"].dropna().astype(str)
    )
    # Public tracker membership identifies prior constituents and subsequent
    # additions, but Nasdaq does not publish the two flags separately.
    selected = set(
        select_annual_companies(
            company,
            previous_constituents=current_companies,
            previous_top_100=current_companies,
        )
    )

    selected_rows = data.loc[
        data["company_id"].isin(selected) & data["regular_eligible"]
    ].copy()
    selected_tickers = tuple(selected_rows["ticker"].drop_duplicates())
    selected_current = set(selected_tickers).intersection(current_tickers)
    additions = tuple(sorted(set(selected_tickers).difference(current_tickers)))
    removals = tuple(sorted(current_tickers.difference(selected_current)))

    data["selected"] = data["ticker"].isin(selected_tickers)
    securities = data.loc[
        data["is_current"] | data["selected"],
        [
            "ticker",
            "company_name",
            "company_id",
            "security_type",
            "is_current",
            "selected",
        ],
    ].drop_duplicates("ticker", keep="first")
    return SelectionResult(
        securities=securities,
        selected_tickers=selected_tickers,
        selected_company_ids=tuple(sorted(selected)),
        additions=additions,
        removals=removals,
        status="calculated",
        source="provided_universe",
        as_of=date.today().isoformat(),
        eligible_company_count=len(company),
        notes=(),
    )


def simulate_rebalance(
    current_holdings: pd.DataFrame,
    reference_data: pd.DataFrame,
    selection: SelectionResult,
    *,
    rebalance_type: str = "quarterly",
) -> RebalanceResult:
    """Calculate post-review weights and their WDI against the same reference."""
    if rebalance_type not in {"quarterly", "annual"}:
        raise ValueError("rebalance_type must be quarterly or annual.")
    current = current_holdings[["ticker", "company_name", "actual_weight"]].copy()
    current["ticker"] = current["ticker"].astype("string").str.upper().str.strip()
    current["actual_weight"] = pd.to_numeric(current["actual_weight"], errors="coerce")
    current = current.loc[current["actual_weight"].fillna(0).gt(0)]
    current["actual_weight"] /= current["actual_weight"].sum()

    securities = selection.securities.copy()
    securities["ticker"] = securities["ticker"].astype("string").str.upper().str.strip()
    data = securities.merge(current, on="ticker", how="outer", suffixes=("", "_current"))
    data["company_name"] = data["company_name"].fillna(data["company_name_current"])
    data["company_id"] = data["company_id"].fillna(
        data["ticker"].map(lambda value: f"TICKER:{value}")
    )
    data["selected"] = data["selected"].fillna(
        data["ticker"].isin(selection.selected_tickers)
    )
    data["is_current"] = data["is_current"].fillna(
        data["ticker"].isin(set(current["ticker"]))
    )
    data["actual_weight"] = data["actual_weight"].fillna(0.0)
    data = data.merge(
        reference_data,
        on="ticker",
        how="left",
        suffixes=("", "_reference"),
    )
    if "security_type_reference" in data:
        data["security_type"] = data["security_type"].fillna(
            data["security_type_reference"]
        )
    for column in [
        "reference_weight_raw",
        "modified_float_mass_raw",
        "counterfactual_reference_raw",
        "float_shares",
        "shares_outstanding",
        "price",
    ]:
        if column not in data:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["modified_float_mass_raw"] = data["modified_float_mass_raw"].fillna(
        data["reference_weight_raw"]
    )
    data["counterfactual_reference_raw"] = data[
        "counterfactual_reference_raw"
    ].fillna(data["reference_weight_raw"])

    calibration = data.loc[
        data["is_current"]
        & data["actual_weight"].gt(0)
        & data["modified_float_mass_raw"].gt(0),
        "modified_float_mass_raw",
    ] / data.loc[
        data["is_current"]
        & data["actual_weight"].gt(0)
        & data["modified_float_mass_raw"].gt(0),
        "actual_weight",
    ]
    current_mass_fallback = (
        data["is_current"]
        & data["selected"]
        & data["actual_weight"].gt(0)
        & ~data["modified_float_mass_raw"].gt(0)
    )
    if not calibration.empty and current_mass_fallback.any():
        data.loc[current_mass_fallback, "modified_float_mass_raw"] = (
            data.loc[current_mass_fallback, "actual_weight"]
            * float(calibration.median())
        )
        data.loc[current_mass_fallback, "rebalance_input_status"] = (
            "current_weight_modified_cap_fallback"
        )

    data["listed_total_cap"] = data["price"] * data["shares_outstanding"]
    conversion_scale, conversion_count = derive_acwi_total_cap_conversion(data)
    reference_source = data.get(
        "reference_source",
        pd.Series(None, index=data.index),
    ).astype("string")
    direct_acwi = (
        reference_source.eq("ishares_acwi").fillna(False)
        & data["modified_float_mass_raw"].gt(0)
    )
    hardcoded_float_reference = reference_source.eq(
        "hardcoded_float_override"
    ).fillna(False)
    fallback_reference = ~direct_acwi & data["modified_float_mass_raw"].gt(0)
    yfinance_float_ratio = data["shares_outstanding"] / data["float_shares"]
    yfinance_ratio_valid = (
        yfinance_float_ratio.replace([np.inf, -np.inf], np.nan)
        .between(1.0, 10.0)
    )

    data["modified_cap_ratio"] = np.nan
    data["modified_market_cap_mass"] = np.nan
    data["rebalance_input_status"] = "missing_modified_cap_mass"

    if conversion_scale is not None:
        converted_total_mass = data["listed_total_cap"] * conversion_scale
        acwi_ratio = (
            converted_total_mass / data["modified_float_mass_raw"]
        ).clip(lower=1.0, upper=3.0)
        direct_valid = direct_acwi & converted_total_mass.gt(0)
        data.loc[direct_valid, "modified_cap_ratio"] = acwi_ratio.loc[direct_valid]
        data.loc[direct_valid, "modified_market_cap_mass"] = (
            data.loc[direct_valid, "modified_float_mass_raw"]
            * acwi_ratio.loc[direct_valid]
        )
        data.loc[direct_valid, "rebalance_input_status"] = (
            "valid_acwi_converted_tso"
        )
        direct_without_total = direct_acwi & ~direct_valid
    else:
        direct_without_total = direct_acwi

    data.loc[direct_without_total, "modified_cap_ratio"] = 1.0
    data.loc[direct_without_total, "modified_market_cap_mass"] = data.loc[
        direct_without_total, "modified_float_mass_raw"
    ]
    data.loc[direct_without_total, "rebalance_input_status"] = (
        "acwi_total_cap_conversion_unavailable_1x"
    )

    fallback_ratio = yfinance_float_ratio.clip(lower=1.0, upper=3.0).where(
        yfinance_ratio_valid,
        1.0,
    )
    data.loc[fallback_reference, "modified_cap_ratio"] = fallback_ratio.loc[
        fallback_reference
    ]
    data.loc[fallback_reference, "modified_market_cap_mass"] = (
        data.loc[fallback_reference, "modified_float_mass_raw"]
        * fallback_ratio.loc[fallback_reference]
    )
    data.loc[
        fallback_reference & yfinance_ratio_valid,
        "rebalance_input_status",
    ] = "valid_yfinance_float_fallback"
    data.loc[
        fallback_reference & ~yfinance_ratio_valid,
        "rebalance_input_status",
    ] = "float_ratio_fallback_1x"
    data.loc[
        fallback_reference
        & hardcoded_float_reference
        & yfinance_ratio_valid,
        "rebalance_input_status",
    ] = "valid_hardcoded_float_override"
    data.loc[
        current_mass_fallback,
        "rebalance_input_status",
    ] = "current_weight_modified_cap_fallback"

    quarterly_existing = (
        rebalance_type == "quarterly"
    ) & data["is_current"] & data["actual_weight"].gt(0)
    weighting_valid = data["selected"] & (
        data["modified_market_cap_mass"].gt(0) | quarterly_existing
    )
    score_valid = weighting_valid & data["counterfactual_reference_raw"].gt(0)
    if not weighting_valid.any():
        raise ValueError("No selected security has a valid rebalance reference mass.")

    valid = data.loc[weighting_valid].copy()
    if rebalance_type == "quarterly":
        security_initial = build_quarterly_index_share_weights(valid)
    else:
        security_initial = valid.set_index("ticker")["modified_market_cap_mass"]
        security_initial = security_initial / security_initial.sum()
    company_initial = (
        valid.assign(initial_weight=valid["ticker"].map(security_initial))
        .groupby("company_id")["initial_weight"]
        .sum()
    )
    quarterly_recap_triggered = (
        rebalance_type == "quarterly"
        and company_capping_required(company_initial)
    )
    if quarterly_recap_triggered:
        raw_valid = valid.loc[valid["modified_market_cap_mass"].gt(0)].copy()
        if raw_valid.empty:
            raise ValueError(
                "Quarterly concentration limits were breached, but no valid "
                "modified-capitalization inputs are available."
            )
        security_initial = raw_valid.set_index("ticker")[
            "modified_market_cap_mass"
        ]
        security_initial = security_initial / security_initial.sum()
        valid = raw_valid
        weighting_valid = data["ticker"].isin(valid["ticker"])
        score_valid = weighting_valid & data["counterfactual_reference_raw"].gt(0)
        company_initial = (
            valid.assign(initial_weight=valid["ticker"].map(security_initial))
            .groupby("company_id")["initial_weight"]
            .sum()
        )
    company_final = apply_company_capping(company_initial)
    company_scale = company_final / company_initial
    security_final = security_initial * valid.set_index("ticker")["company_id"].map(
        company_scale
    )
    security_final = security_final / security_final.sum()
    if rebalance_type == "annual":
        security_final = apply_annual_security_capping(security_final)

    score_tickers = data.loc[score_valid, "ticker"]
    score_weights = security_final.reindex(score_tickers).dropna()
    score_weights = score_weights / score_weights.sum()
    reference = data.loc[score_valid].set_index("ticker")[
        "counterfactual_reference_raw"
    ]
    reference = reference / reference.sum()
    data["rebalance_weight"] = data["ticker"].map(security_final).fillna(0.0)
    data["rebalance_reference_weight"] = data["ticker"].map(reference)
    data["rebalance_weight_change"] = (
        data["rebalance_weight"] - data["actual_weight"]
    )
    data["rebalance_weight_delta"] = data["ticker"].map(
        score_weights - reference
    )
    data["rebalance_distortion_contribution"] = (
        50.0 * data["rebalance_weight_delta"].abs()
    )
    data["rebalance_membership"] = data["selected"].astype(bool)

    current_selected_weight = float(
        data.loc[data["is_current"] & data["selected"], "actual_weight"].sum()
    )
    valid_current_selected_weight = float(
        data.loc[
            data["is_current"] & score_valid,
            "actual_weight",
        ].sum()
    )
    coverage = (
        valid_current_selected_weight / current_selected_weight
        if current_selected_weight > 0
        else 1.0
    )
    score = float(data.loc[score_valid, "rebalance_distortion_contribution"].sum())
    input_fallbacks = int(
        data.loc[weighting_valid, "rebalance_input_status"]
        .isin(
            [
                "float_ratio_fallback_1x",
                "current_weight_modified_cap_fallback",
                "acwi_total_cap_conversion_unavailable_1x",
            ]
        )
        .sum()
    )
    notes = list(selection.notes)
    if conversion_scale is not None:
        notes.append(
            "Direct ACWI modified caps use an ACWI/total-cap conversion scale "
            f"of {conversion_scale:.8g}, calibrated on {conversion_count} "
            "matched securities; yfinance floatShares is not used for them."
        )
    hardcoded_float_count = int(
        (
            weighting_valid
            & data["rebalance_input_status"].eq(
                "valid_hardcoded_float_override"
            )
        ).sum()
    )
    if hardcoded_float_count:
        notes.append(
            f"{hardcoded_float_count} security used a maintained "
            "listing-specific float-share override instead of yfinance."
        )
    if rebalance_type == "quarterly":
        notes.append(
            "Quarterly initial weights preserve current published index weights "
            "as a public proxy for Nasdaq Index Shares. Exact accumulated TSO "
            "changes since the previous update require proprietary prior Index "
            "Shares and are not independently reproducible."
        )
        if quarterly_recap_triggered:
            notes.append(
                "The quarterly 24%/48% concentration test was breached, so the "
                "company-level adjustment was recalculated from modified market "
                "capitalization."
            )
        else:
            notes.append(
                "The quarterly 24%/48% concentration test was not breached; no "
                "40% cohort redistribution was applied."
            )
    if input_fallbacks:
        notes.append(
            f"{input_fallbacks} securities used a documented modified-cap fallback "
            "because a public float/TSO input was unavailable or inconsistent."
        )
    status = (
        selection.status
        if coverage >= 0.99
        else f"{selection.status}_partial_coverage"
    )
    return RebalanceResult(
        components=data,
        ndx_wdi=score,
        coverage_ratio=coverage,
        constituent_count=int(weighting_valid.sum()),
        status=status,
        method=(
            "quarterly_index_shares_then_modified_cap_2026"
            if rebalance_type == "quarterly"
            else "annual_modified_market_cap_2026"
        ),
        reference_date=selection.as_of,
        additions=selection.additions,
        removals=selection.removals,
        data_source=selection.source,
        acwi_conversion_scale=conversion_scale,
        acwi_calibration_count=conversion_count,
        notes=tuple(notes),
    )


def derive_acwi_total_cap_conversion(
    data: pd.DataFrame,
    *,
    quantile: float = 0.90,
    minimum_pairs: int = 20,
) -> tuple[float | None, int]:
    """Estimate fund-value units per total-cap dollar from direct ACWI matches.

    ACWI holding market values are proportional to free-float capitalization.
    The upper quantile of ACWI mass / listed total capitalization estimates the
    common conversion scale from names whose free float is close to 100%.
    """
    if not 0.5 <= quantile < 1.0:
        raise ValueError("quantile must be between 0.5 and 1.0.")
    reference_source = data.get(
        "reference_source",
        pd.Series(None, index=data.index),
    ).astype("string")
    float_mass = pd.to_numeric(
        data.get("modified_float_mass_raw"), errors="coerce"
    )
    listed_total_cap = pd.to_numeric(
        data.get("listed_total_cap"), errors="coerce"
    )
    valid = (
        reference_source.eq("ishares_acwi")
        & float_mass.gt(0)
        & listed_total_cap.gt(0)
    )
    ratios = (float_mass.loc[valid] / listed_total_cap.loc[valid]).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(ratios) < minimum_pairs:
        return None, int(len(ratios))
    return float(ratios.quantile(quantile)), int(len(ratios))


def build_quarterly_index_share_weights(data: pd.DataFrame) -> pd.Series:
    """Build quarterly initial weights from inherited Index Shares.

    Published tracker weights proxy current price times Nasdaq Index Shares.
    New constituents are inserted by Nasdaq's linear weight interpolation rule
    using their modified-capitalization rank.
    """
    indexed = data.set_index("ticker", drop=False)
    existing = indexed.loc[
        indexed["is_current"].fillna(False).astype(bool)
        & indexed["actual_weight"].gt(0)
    ]
    weights = existing["actual_weight"].astype(float).copy()
    masses = pd.to_numeric(
        indexed["modified_market_cap_mass"], errors="coerce"
    )

    additions = indexed.loc[
        ~indexed.index.isin(weights.index) & masses.gt(0)
    ].sort_values("modified_market_cap_mass", ascending=False)
    for ticker, row in additions.iterrows():
        mass = float(row["modified_market_cap_mass"])
        peers = pd.DataFrame(
            {
                "mass": masses.reindex(weights.index),
                "weight": weights,
            }
        ).dropna()
        larger = peers.loc[peers["mass"].gt(mass)].sort_values("mass").head(1)
        smaller = peers.loc[peers["mass"].lt(mass)].sort_values(
            "mass", ascending=False
        ).head(1)

        if not larger.empty and not smaller.empty:
            x1, y1 = map(float, larger.iloc[0][["mass", "weight"]])
            x0, y0 = map(float, smaller.iloc[0][["mass", "weight"]])
            interpolated = y0 + (mass - x0) * (y1 - y0) / (x1 - x0)
        elif not larger.empty:
            x1, y1 = map(float, larger.iloc[0][["mass", "weight"]])
            interpolated = y1 * mass / x1
        elif not smaller.empty:
            x0, y0 = map(float, smaller.iloc[0][["mass", "weight"]])
            interpolated = y0 * mass / x0
        else:
            interpolated = mass
        weights.loc[ticker] = max(float(interpolated), np.finfo(float).eps)

    if weights.empty:
        fallback = masses.loc[masses.gt(0)]
        if fallback.empty:
            raise ValueError("No quarterly Index Share proxy is available.")
        weights = fallback
    return weights / weights.sum()


def company_capping_required(initial_weights: pd.Series) -> bool:
    """Return whether quarterly company concentration adjustment is required."""
    weights = _normalize_positive(initial_weights)
    if weights.gt(0.24 + 1e-12).any():
        return True
    return float(weights.loc[weights.gt(0.045 + 1e-12)].sum()) >= 0.48 - 1e-12


def apply_company_capping(initial_weights: pd.Series) -> pd.Series:
    """Apply the 2026 company-level 20% and aggregate 40% adjustments."""
    weights = _normalize_positive(initial_weights)
    # The official constraints assume the approximately 100-company NDX
    # universe. Tiny synthetic fixtures cannot mathematically absorb a 20% cap.
    if len(weights) < 5:
        return weights

    for _ in range(len(weights) + 1):
        changed = False
        if weights.gt(0.24 + 1e-12).any():
            weights = _proportional_with_caps(
                weights,
                pd.Series(0.20, index=weights.index),
            )
            changed = True

        stage_two_base = weights.copy()
        large = set(weights.index[weights.gt(0.045 + 1e-12)])
        if float(weights.loc[list(large)].sum()) >= 0.48 - 1e-12:
            large_index = weights.index.isin(large)
            large_base = stage_two_base.loc[large_index]
            small_base = stage_two_base.loc[~large_index]
            if small_base.empty:
                raise ValueError(
                    "Company-level aggregate capping requires companies outside "
                    "the greater-than-4.5% cohort."
                )
            adjusted = pd.Series(0.0, index=weights.index)
            adjusted.loc[large_index] = 0.40 * large_base / large_base.sum()
            smallest_large = float(adjusted.loc[large_index].min())
            outside_cap = min(0.045, smallest_large)
            adjusted.loc[~large_index] = _proportional_with_caps(
                small_base / small_base.sum(),
                pd.Series(outside_cap / 0.60, index=small_base.index),
            ) * 0.60
            weights = adjusted
            changed = True

        if not company_capping_required(weights):
            break
        if not changed:
            break
    return weights / weights.sum()


def apply_annual_security_capping(initial_weights: pd.Series) -> pd.Series:
    """Apply the annual security-level 14%, top-five and 4.4% constraints."""
    weights = _normalize_positive(initial_weights)
    # At least eight securities are needed to absorb a universal 14% cap.
    # This guard only affects deliberately tiny synthetic test fixtures.
    if len(weights) < 8:
        return weights
    for _ in range(len(weights) + 1):
        if not weights.gt(0.15 + 1e-12).any():
            break
        base = weights.copy()
        caps = pd.Series(1.0, index=weights.index)
        caps.loc[weights.gt(0.15 + 1e-12)] = 0.14
        weights = _proportional_with_caps(base, caps)

    for _ in range(len(weights) + 1):
        top_five = list(weights.nlargest(min(5, len(weights))).index)
        if (
            len(top_five) < 5
            or float(weights.loc[top_five].sum()) < 0.40 - 1e-12
        ):
            break
        stage_two_base = weights.copy()
        top_base = stage_two_base.loc[top_five]
        outside = stage_two_base.drop(index=top_five)
        result = pd.Series(0.0, index=weights.index)
        result.loc[top_five] = 0.385 * top_base / top_base.sum()
        outside_cap = min(0.044, float(result.loc[top_five].min()))
        if not outside.empty:
            result.loc[outside.index] = _proportional_with_caps(
                outside / outside.sum(),
                pd.Series(outside_cap / 0.615, index=outside.index),
            ) * 0.615
        weights = result
    return weights / weights.sum()


def _normalize_positive(weights: pd.Series) -> pd.Series:
    result = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    result = result.loc[result.gt(0)]
    if result.empty:
        raise ValueError("At least one positive weight is required.")
    return result / result.sum()


def _proportional_with_caps(base: pd.Series, caps: pd.Series) -> pd.Series:
    """Solve weight_i=min(cap_i, adjustment_factor*base_i), summing to one."""
    base = _normalize_positive(base)
    caps = caps.reindex(base.index).fillna(1.0).astype(float)
    if float(caps.sum()) < 1.0 - 1e-12:
        raise ValueError("Capping constraints cannot sum to 100%.")
    low, high = 0.0, 1.0
    while float(np.minimum(caps, high * base).sum()) < 1.0:
        high *= 2.0
    for _ in range(100):
        middle = (low + high) / 2.0
        total = float(np.minimum(caps, middle * base).sum())
        if total < 1.0:
            low = middle
        else:
            high = middle
    result = np.minimum(caps, high * base)
    return result / result.sum()
