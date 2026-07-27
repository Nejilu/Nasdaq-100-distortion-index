"""FastAPI surface for NDX-WDI snapshots."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from database import SnapshotDatabase
from snapshot_service import recompute_all_snapshots, recompute_snapshot


load_dotenv()
app = FastAPI(
    title="NDX Weight Distortion Index API",
    version="0.2.0",
    description=(
        "Compare published ETF weights with ACWI free-float or total-capitalization "
        "counterfactual weights."
    ),
)


class RecomputeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    universe: Literal["all", "non_ucits", "ucits"] = "all"
    weighting_basis: Literal["float", "total"] = "float"
    holdings_csv: str | None = None


@lru_cache(maxsize=8)
def _database_for_path(path: str) -> SnapshotDatabase:
    return SnapshotDatabase(path)


def get_database() -> SnapshotDatabase:
    return _database_for_path(os.getenv("NDX_DB_PATH", "data/ndx_wdi.sqlite3"))


@app.get("/api/current")
def current(
    universe: Literal["non_ucits", "ucits"] | None = None,
    weighting_basis: Literal["float", "total"] = "float",
) -> dict[str, object]:
    database = get_database()
    if universe:
        snapshot = database.get_current(universe, weighting_basis)
        if snapshot is None:
            raise HTTPException(status_code=404, detail=f"No {universe} snapshot is available.")
        snapshot["snapshot_status"] = snapshot["status"]
        return snapshot
    snapshots = database.get_current_by_universe(weighting_basis)
    if not snapshots:
        raise HTTPException(status_code=404, detail="No snapshots are available.")
    for snapshot in snapshots.values():
        snapshot["snapshot_status"] = snapshot["status"]
    return {"snapshots": snapshots}


@app.get("/api/history")
def history(
    limit: int = Query(365, ge=1, le=5000),
    universe: Literal["non_ucits", "ucits"] | None = None,
    weighting_basis: Literal["float", "total"] = "float",
) -> list[dict[str, object]]:
    rows = get_database().get_history(
        limit=limit, universe=universe, weighting_basis=weighting_basis
    )
    for row in rows:
        row["snapshot_status"] = row["status"]
    return rows


@app.get("/api/components")
def components(
    snapshot_id: int | None = Query(None, ge=1),
    universe: Literal["non_ucits", "ucits"] | None = None,
    weighting_basis: Literal["float", "total"] = "float",
    ranking: Literal["all", "overweights", "underweights", "contributors"] = "all",
    limit: int = Query(500, ge=1, le=1000),
) -> list[dict[str, object]]:
    database = get_database()
    if snapshot_id is not None and database.get_snapshot(snapshot_id) is None:
        raise HTTPException(status_code=404, detail="Snapshot not found.")
    rows = database.get_components(
        snapshot_id, universe=universe, weighting_basis=weighting_basis
    )
    valid = [row for row in rows if str(row["data_status"]).startswith("valid")]
    if ranking == "overweights":
        rows = sorted(
            (row for row in valid if row["weight_delta"] > 0),
            key=lambda row: row["weight_delta"],
            reverse=True,
        )
    elif ranking == "underweights":
        rows = sorted(
            (row for row in valid if row["weight_delta"] < 0),
            key=lambda row: row["weight_delta"],
        )
    elif ranking == "contributors":
        rows = sorted(valid, key=lambda row: row["distortion_contribution"], reverse=True)
    return rows[:limit]


@app.get("/api/active-share")
def active_share(
    snapshot_id: int | None = Query(None, ge=1),
    universe: Literal["non_ucits", "ucits"] = "non_ucits",
    weighting_basis: Literal["float", "total"] = "float",
    ranking: Literal[
        "all",
        "ndx_overweights",
        "spx_overweights",
        "contributors",
    ] = "all",
    rebalanced: bool = False,
    limit: int = Query(1000, ge=1, le=1000),
) -> dict[str, object]:
    database = get_database()
    if snapshot_id is not None and database.get_snapshot(snapshot_id) is None:
        raise HTTPException(status_code=404, detail="Snapshot not found.")
    summary = database.get_active_share(
        snapshot_id,
        universe=universe,
        weighting_basis=weighting_basis,
    )
    if summary is None:
        raise HTTPException(
            status_code=404,
            detail="No Active Share comparison is available.",
        )
    rows = database.get_active_share_components(
        int(summary["snapshot_id"]),
    )
    delta_column = (
        "rebalanced_weight_delta" if rebalanced else "weight_delta"
    )
    if ranking == "ndx_overweights":
        rows = sorted(
            (row for row in rows if (row.get(delta_column) or 0) > 0),
            key=lambda row: row.get(delta_column) or 0,
            reverse=True,
        )
    elif ranking == "spx_overweights":
        rows = sorted(
            (row for row in rows if (row.get(delta_column) or 0) < 0),
            key=lambda row: row.get(delta_column) or 0,
        )
    elif ranking == "contributors":
        rows = sorted(
            rows,
            key=lambda row: abs(row.get(delta_column) or 0),
            reverse=True,
        )
    return {"summary": summary, "components": rows[:limit]}


@app.post("/api/recompute", status_code=201)
def recompute(payload: RecomputeRequest) -> dict[str, object]:
    if payload.holdings_csv and not Path(payload.holdings_csv).exists():
        raise HTTPException(status_code=400, detail="The holdings_csv file does not exist.")
    try:
        if payload.universe == "all":
            if payload.holdings_csv:
                raise HTTPException(
                    status_code=400,
                    detail="holdings_csv requires universe=non_ucits or universe=ucits.",
                )
            outcomes = recompute_all_snapshots(
                db_path=os.getenv("NDX_DB_PATH", "data/ndx_wdi.sqlite3"),
                weighting_basis=payload.weighting_basis,
            )
            return {"snapshots": {outcome.universe: outcome.summary() for outcome in outcomes}}
        return recompute_snapshot(
            db_path=os.getenv("NDX_DB_PATH", "data/ndx_wdi.sqlite3"),
            holdings_csv=payload.holdings_csv,
            universe=payload.universe,
            weighting_basis=payload.weighting_basis,
        ).summary()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Recomputation failed: {exc}") from exc
