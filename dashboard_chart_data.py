"""Pure data preparation helpers for dashboard figures."""

from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_constituent_weight_comparison(
    components: pd.DataFrame,
    *,
    limit: int,
) -> pd.DataFrame:
    """Split current and counterfactual weights into visual bar segments."""
    required = {
        "ticker",
        "actual_weight",
        "counterfactual_weight",
        "data_status",
    }
    missing = required.difference(components.columns)
    if missing:
        raise ValueError(
            "Constituent comparison is missing columns: "
            f"{sorted(missing)}"
        )
    if limit <= 0:
        raise ValueError("limit must be positive.")

    data = components.copy()
    if "company_name" not in data:
        data["company_name"] = data["ticker"]
    data["actual_weight"] = pd.to_numeric(
        data["actual_weight"],
        errors="coerce",
    )
    data["counterfactual_weight"] = pd.to_numeric(
        data["counterfactual_weight"],
        errors="coerce",
    )
    valid = data["data_status"].astype("string").str.startswith(
        "valid",
        na=False,
    )
    data = data.loc[
        valid
        & data["actual_weight"].gt(0)
        & data["counterfactual_weight"].ge(0)
    ].copy()
    if data.empty:
        return data

    data = data.nlargest(min(limit, len(data)), "actual_weight").copy()
    data["shared_weight"] = data[
        ["actual_weight", "counterfactual_weight"]
    ].min(axis=1)
    data["actual_excess"] = (
        data["actual_weight"] - data["counterfactual_weight"]
    ).clip(lower=0)
    data["counterfactual_gap"] = (
        data["counterfactual_weight"] - data["actual_weight"]
    ).clip(lower=0)
    data["display_weight"] = data[
        ["actual_weight", "counterfactual_weight"]
    ].max(axis=1)
    data["weight_delta"] = (
        data["actual_weight"] - data["counterfactual_weight"]
    )
    data["counterfactual_label_in_shared"] = np.where(
        data["actual_weight"].ge(data["counterfactual_weight"]),
        data["counterfactual_weight"].map(lambda value: f"CF {value:.2%}"),
        "",
    )
    data["counterfactual_label_in_gap"] = np.where(
        data["counterfactual_weight"].gt(data["actual_weight"]),
        data["counterfactual_weight"].map(lambda value: f"CF {value:.2%}"),
        "",
    )
    return data.sort_values("actual_weight", ascending=True).reset_index(
        drop=True
    )
