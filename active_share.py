"""Active Share calculation between Nasdaq-100 and S&P 500 ETF holdings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ActiveShareResult:
    active_share: float
    rebalanced_active_share: float | None
    components: pd.DataFrame
    spx_reference_fund: str
    spx_holdings_source: str
    spx_holdings_as_of: str | None = None
    status: str = "complete"


@dataclass(frozen=True)
class ActiveShareSleeves:
    components: pd.DataFrame
    ndx_active_mass: float
    spx_active_mass: float
    overlap_mass: float


def calculate_active_share(
    ndx_components: pd.DataFrame,
    spx_holdings: pd.DataFrame,
    *,
    spx_reference_fund: str,
    spx_holdings_source: str,
    spx_holdings_as_of: str | None = None,
) -> ActiveShareResult:
    """Calculate Active Share on the full union of NDX and SPX securities."""
    required_ndx = {"ticker", "actual_weight"}
    required_spx = {"ticker", "actual_weight"}
    if not required_ndx.issubset(ndx_components.columns):
        raise ValueError(
            "NDX components are missing: "
            f"{sorted(required_ndx.difference(ndx_components.columns))}"
        )
    if not required_spx.issubset(spx_holdings.columns):
        raise ValueError(
            "SPX holdings are missing: "
            f"{sorted(required_spx.difference(spx_holdings.columns))}"
        )

    ndx = _normalized_holdings(
        ndx_components,
        weight_column="actual_weight",
        output_column="ndx_weight",
    )
    spx = _normalized_holdings(
        spx_holdings,
        weight_column="actual_weight",
        output_column="spx_weight",
    )
    components = ndx.merge(
        spx,
        on="ticker",
        how="outer",
        suffixes=("_ndx", "_spx"),
    )
    components["company_name"] = components["company_name_ndx"].fillna(
        components["company_name_spx"]
    )
    components["ndx_weight"] = components["ndx_weight"].fillna(0.0)
    components["spx_weight"] = components["spx_weight"].fillna(0.0)
    components["weight_delta"] = (
        components["ndx_weight"] - components["spx_weight"]
    )
    components["absolute_delta"] = components["weight_delta"].abs()

    rebalanced_active_share: float | None = None
    if "rebalance_weight" in ndx_components:
        rebalanced = _normalized_holdings(
            ndx_components,
            weight_column="rebalance_weight",
            output_column="rebalanced_ndx_weight",
            allow_empty=True,
        )[["ticker", "rebalanced_ndx_weight"]]
        components = components.merge(rebalanced, on="ticker", how="left")
        if not rebalanced.empty:
            components["rebalanced_ndx_weight"] = components[
                "rebalanced_ndx_weight"
            ].fillna(0.0)
            components["rebalanced_weight_delta"] = (
                components["rebalanced_ndx_weight"] - components["spx_weight"]
            )
            rebalanced_active_share = 0.5 * float(
                components["rebalanced_weight_delta"].abs().sum()
            )
        else:
            components["rebalanced_ndx_weight"] = np.nan
            components["rebalanced_weight_delta"] = np.nan
    else:
        components["rebalanced_ndx_weight"] = np.nan
        components["rebalanced_weight_delta"] = np.nan

    active_share = 0.5 * float(components["absolute_delta"].sum())
    columns = [
        "ticker",
        "company_name",
        "ndx_weight",
        "spx_weight",
        "weight_delta",
        "absolute_delta",
        "rebalanced_ndx_weight",
        "rebalanced_weight_delta",
    ]
    components = components[columns].sort_values(
        ["absolute_delta", "ticker"],
        ascending=[False, True],
    )
    return ActiveShareResult(
        active_share=active_share,
        rebalanced_active_share=rebalanced_active_share,
        components=components.reset_index(drop=True),
        spx_reference_fund=spx_reference_fund,
        spx_holdings_source=spx_holdings_source,
        spx_holdings_as_of=spx_holdings_as_of,
    )


def calculate_active_share_sleeves(
    components: pd.DataFrame,
    *,
    ndx_weight_column: str = "ndx_weight",
) -> ActiveShareSleeves:
    """Build NDX-active, SPX-active, and overlap portfolios normalized to 100%."""
    required = {"ticker", ndx_weight_column, "spx_weight"}
    missing = required.difference(components.columns)
    if missing:
        raise ValueError(f"Active Share components are missing: {sorted(missing)}")

    frame = pd.DataFrame(
        {
            "ticker": components["ticker"].map(_canonical_ticker),
            "company_name": components.get(
                "company_name",
                components["ticker"],
            ),
            "ndx_weight": pd.to_numeric(
                components[ndx_weight_column],
                errors="coerce",
            ),
            "spx_weight": pd.to_numeric(
                components["spx_weight"],
                errors="coerce",
            ),
        }
    ).dropna(subset=["ndx_weight", "spx_weight"])
    if (frame[["ndx_weight", "spx_weight"]] < 0).any().any():
        raise ValueError("Active Share sleeve weights cannot be negative.")

    frame = (
        frame.groupby("ticker", as_index=False, sort=False)
        .agg(
            company_name=("company_name", "first"),
            ndx_weight=("ndx_weight", "sum"),
            spx_weight=("spx_weight", "sum"),
        )
    )
    ndx_total = float(frame["ndx_weight"].sum())
    spx_total = float(frame["spx_weight"].sum())
    if ndx_total <= 0 or spx_total <= 0:
        raise ValueError("Both portfolios need positive holdings weights.")
    frame["ndx_weight"] /= ndx_total
    frame["spx_weight"] /= spx_total

    frame["ndx_active_raw"] = (
        frame["ndx_weight"] - frame["spx_weight"]
    ).clip(lower=0)
    frame["spx_active_raw"] = (
        frame["spx_weight"] - frame["ndx_weight"]
    ).clip(lower=0)
    frame["overlap_raw"] = frame[["ndx_weight", "spx_weight"]].min(axis=1)

    masses = {
        "ndx_active": float(frame["ndx_active_raw"].sum()),
        "spx_active": float(frame["spx_active_raw"].sum()),
        "overlap": float(frame["overlap_raw"].sum()),
    }
    for sleeve in ["ndx_active", "spx_active", "overlap"]:
        mass = masses[sleeve]
        frame[f"{sleeve}_weight"] = (
            frame[f"{sleeve}_raw"] / mass if mass > 0 else 0.0
        )

    return ActiveShareSleeves(
        components=frame,
        ndx_active_mass=masses["ndx_active"],
        spx_active_mass=masses["spx_active"],
        overlap_mass=masses["overlap"],
    )


def _normalized_holdings(
    frame: pd.DataFrame,
    *,
    weight_column: str,
    output_column: str,
    allow_empty: bool = False,
) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "ticker": frame["ticker"].map(_canonical_ticker),
            "company_name": frame.get(
                "company_name",
                frame["ticker"],
            ),
            output_column: pd.to_numeric(
                frame.get(weight_column),
                errors="coerce",
            ),
        }
    )
    data = data.loc[
        data["ticker"].ne("")
        & data[output_column].notna()
        & data[output_column].ge(0)
    ].copy()
    data = (
        data.groupby("ticker", as_index=False, sort=False)
        .agg(
            company_name=("company_name", "first"),
            **{output_column: (output_column, "sum")},
        )
    )
    total = float(data[output_column].sum())
    if total <= 0:
        if allow_empty:
            return data.iloc[0:0]
        raise ValueError(f"{weight_column} contains no positive holdings weights.")
    data[output_column] = data[output_column] / total
    return data


def _canonical_ticker(value: object) -> str:
    ticker = str(value).upper().strip()
    aliases = {
        "BRK.B": "BRKB",
        "BRK/B": "BRKB",
        "BRK-B": "BRKB",
        "BF.B": "BFB",
        "BF/B": "BFB",
        "BF-B": "BFB",
    }
    return aliases.get(ticker, ticker)
