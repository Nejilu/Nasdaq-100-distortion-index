"""Shared market-data quality rules used by the calculation engines."""

from __future__ import annotations

import pandas as pd


def evaluate_float_observations(
    frame: pd.DataFrame,
    *,
    float_shares_tolerance: float = 1.10,
    float_cap_tolerance: float = 1.25,
) -> pd.DataFrame:
    """Return validity masks for float-based calculations."""
    data = frame.copy()
    for column in ["price", "float_shares", "shares_outstanding", "market_cap"]:
        if column not in data:
            data[column] = pd.NA
        data[column] = pd.to_numeric(data[column], errors="coerce")

    quality = pd.DataFrame(index=data.index)
    quality["price_valid"] = data["price"].notna() & data["price"].gt(0)
    quality["float_valid"] = (
        data["float_shares"].notna() & data["float_shares"].gt(0)
    )
    quality["outstanding_valid"] = (
        data["shares_outstanding"].notna()
        & data["shares_outstanding"].gt(0)
    )
    quality["market_cap_valid"] = (
        data["market_cap"].notna() & data["market_cap"].gt(0)
    )
    quality["float_cap"] = data["price"] * data["float_shares"]
    quality["inconsistent"] = (
        quality["float_valid"]
        & quality["outstanding_valid"]
        & (
            data["float_shares"]
            > data["shares_outstanding"] * float_shares_tolerance
        )
    ) | (
        quality["price_valid"]
        & quality["float_valid"]
        & quality["market_cap_valid"]
        & (
            quality["float_cap"]
            > data["market_cap"] * float_cap_tolerance
        )
    )
    quality["valid"] = (
        quality["price_valid"]
        & quality["float_valid"]
        & ~quality["inconsistent"]
    )
    return quality
