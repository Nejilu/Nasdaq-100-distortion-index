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
