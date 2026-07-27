"""SQLite persistence for NDX-WDI snapshots."""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from distortion_engine import DistortionResult
from nasdaq100_rebalance import RebalanceResult


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    ndx_wdi REAL NOT NULL,
    coverage_ratio REAL NOT NULL,
    constituent_count INTEGER NOT NULL,
    missing_float_count INTEGER NOT NULL,
    invalid_float_count INTEGER NOT NULL DEFAULT 0,
    missing_reference_shares_count INTEGER NOT NULL DEFAULT 0,
    missing_price_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    weighting_basis TEXT NOT NULL DEFAULT 'float',
    universe TEXT NOT NULL DEFAULT 'non_ucits',
    reference_fund TEXT,
    holdings_as_of TEXT,
    reference_data_as_of TEXT,
    source_failures TEXT,
    holdings_source TEXT,
    market_data_source TEXT
    ,rebalance_ndx_wdi REAL
    ,rebalance_coverage_ratio REAL
    ,rebalance_constituent_count INTEGER
    ,rebalance_status TEXT
    ,rebalance_method TEXT
    ,rebalance_reference_date TEXT
    ,rebalance_additions TEXT
    ,rebalance_removals TEXT
    ,rebalance_data_source TEXT
    ,rebalance_notes TEXT
);

CREATE TABLE IF NOT EXISTS snapshot_components (
    snapshot_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    company_name TEXT,
    actual_weight REAL,
    float_weight REAL,
    counterfactual_weight REAL,
    weight_delta REAL,
    weight_ratio REAL,
    distortion_contribution REAL,
    price REAL,
    float_shares REAL,
    reference_shares REAL,
    security_type TEXT NOT NULL DEFAULT 'Ordinary share',
    reference_source TEXT,
    acwi_weight REAL,
    acwi_listing TEXT,
    data_status TEXT NOT NULL DEFAULT 'valid',
    rebalance_weight REAL,
    rebalance_reference_weight REAL,
    rebalance_weight_change REAL,
    rebalance_weight_delta REAL,
    rebalance_distortion_contribution REAL,
    rebalance_membership INTEGER,
    rebalance_company_id TEXT,
    modified_cap_ratio REAL,
    modified_market_cap_mass REAL,
    rebalance_input_status TEXT,
    PRIMARY KEY (snapshot_id, ticker),
    FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON snapshots(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_components_ticker ON snapshot_components(ticker);
"""


class SnapshotDatabase:
    def __init__(self, path: str | Path = "data/ndx_wdi.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            self._ensure_snapshot_columns(connection)
            self._ensure_component_columns(connection)
            connection.execute(
                "UPDATE snapshots SET missing_reference_shares_count = missing_float_count "
                "WHERE weighting_basis = 'float' AND missing_reference_shares_count = 0"
            )
            connection.execute(
                "UPDATE snapshot_components SET counterfactual_weight = float_weight "
                "WHERE counterfactual_weight IS NULL AND float_weight IS NOT NULL"
            )
            connection.execute(
                "UPDATE snapshot_components SET reference_shares = float_shares "
                "WHERE reference_shares IS NULL AND float_shares IS NOT NULL"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_snapshots_universe_timestamp "
                "ON snapshots(universe, timestamp DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_snapshots_basis_universe_timestamp "
                "ON snapshots(weighting_basis, universe, timestamp DESC)"
            )
            connection.commit()

    @staticmethod
    def _ensure_snapshot_columns(connection: sqlite3.Connection) -> None:
        existing = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(snapshots)").fetchall()
        }
        additions = {
            "invalid_float_count": "INTEGER NOT NULL DEFAULT 0",
            "missing_reference_shares_count": "INTEGER NOT NULL DEFAULT 0",
            "weighting_basis": "TEXT NOT NULL DEFAULT 'float'",
            "universe": "TEXT NOT NULL DEFAULT 'non_ucits'",
            "reference_fund": "TEXT",
            "holdings_as_of": "TEXT",
            "reference_data_as_of": "TEXT",
            "source_failures": "TEXT",
            "rebalance_ndx_wdi": "REAL",
            "rebalance_coverage_ratio": "REAL",
            "rebalance_constituent_count": "INTEGER",
            "rebalance_status": "TEXT",
            "rebalance_method": "TEXT",
            "rebalance_reference_date": "TEXT",
            "rebalance_additions": "TEXT",
            "rebalance_removals": "TEXT",
            "rebalance_data_source": "TEXT",
            "rebalance_notes": "TEXT",
        }
        for column, definition in additions.items():
            if column not in existing:
                connection.execute(f"ALTER TABLE snapshots ADD COLUMN {column} {definition}")

    @staticmethod
    def _ensure_component_columns(connection: sqlite3.Connection) -> None:
        existing = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(snapshot_components)").fetchall()
        }
        additions = {
            "counterfactual_weight": "REAL",
            "reference_shares": "REAL",
            "security_type": "TEXT NOT NULL DEFAULT 'Ordinary share'",
            "reference_source": "TEXT",
            "acwi_weight": "REAL",
            "acwi_listing": "TEXT",
            "rebalance_weight": "REAL",
            "rebalance_reference_weight": "REAL",
            "rebalance_weight_change": "REAL",
            "rebalance_weight_delta": "REAL",
            "rebalance_distortion_contribution": "REAL",
            "rebalance_membership": "INTEGER",
            "rebalance_company_id": "TEXT",
            "modified_cap_ratio": "REAL",
            "modified_market_cap_mass": "REAL",
            "rebalance_input_status": "TEXT",
        }
        for column, definition in additions.items():
            if column not in existing:
                connection.execute(
                    f"ALTER TABLE snapshot_components ADD COLUMN {column} {definition}"
                )

    def save_snapshot(
        self,
        result: DistortionResult,
        *,
        timestamp: str,
        holdings_source: str,
        market_data_source: str,
        universe: str = "non_ucits",
        weighting_basis: str = "float",
        reference_fund: str | None = None,
        holdings_as_of: str | None = None,
        reference_data_as_of: str | None = None,
        source_failures: str | None = None,
        rebalance: RebalanceResult | None = None,
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO snapshots (
                    timestamp, ndx_wdi, coverage_ratio, constituent_count,
                    missing_float_count, invalid_float_count, missing_reference_shares_count,
                    missing_price_count, status, weighting_basis, universe,
                    reference_fund, holdings_as_of, reference_data_as_of, source_failures,
                    holdings_source, market_data_source, rebalance_ndx_wdi,
                    rebalance_coverage_ratio, rebalance_constituent_count,
                    rebalance_status, rebalance_method, rebalance_reference_date,
                    rebalance_additions, rebalance_removals, rebalance_data_source,
                    rebalance_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    result.ndx_wdi,
                    result.coverage_ratio,
                    result.constituent_count,
                    result.missing_float_count,
                    result.invalid_float_count,
                    result.missing_reference_shares_count,
                    result.missing_price_count,
                    result.snapshot_status,
                    weighting_basis,
                    universe,
                    reference_fund,
                    holdings_as_of,
                    reference_data_as_of,
                    source_failures,
                    holdings_source,
                    market_data_source,
                    _sql_value(rebalance.ndx_wdi) if rebalance else None,
                    _sql_value(rebalance.coverage_ratio) if rebalance else None,
                    rebalance.constituent_count if rebalance else None,
                    rebalance.status if rebalance else None,
                    rebalance.method if rebalance else None,
                    rebalance.reference_date if rebalance else None,
                    json.dumps(rebalance.additions) if rebalance else None,
                    json.dumps(rebalance.removals) if rebalance else None,
                    rebalance.data_source if rebalance else None,
                    json.dumps(rebalance.notes) if rebalance else None,
                ),
            )
            snapshot_id = int(cursor.lastrowid)
            rows = []
            for component in result.components.to_dict(orient="records"):
                rows.append(
                    (
                        snapshot_id,
                        component["ticker"],
                        component.get("company_name"),
                        _sql_value(component.get("actual_weight")),
                        _sql_value(component.get("float_weight")),
                        _sql_value(component.get("counterfactual_weight")),
                        _sql_value(component.get("weight_delta")),
                        _sql_value(component.get("weight_ratio")),
                        _sql_value(component.get("distortion_contribution")),
                        _sql_value(component.get("price")),
                        _sql_value(component.get("float_shares")),
                        _sql_value(component.get("reference_shares")),
                        (
                            "Ordinary share"
                            if pd.isna(component.get("security_type"))
                            else component.get("security_type")
                        ),
                        component.get("reference_source"),
                        _sql_value(component.get("acwi_weight")),
                        component.get("acwi_listing"),
                        (
                            "valid"
                            if pd.isna(component.get("data_status"))
                            else component.get("data_status")
                        ),
                        _sql_value(component.get("rebalance_weight")),
                        _sql_value(component.get("rebalance_reference_weight")),
                        _sql_value(component.get("rebalance_weight_change")),
                        _sql_value(component.get("rebalance_weight_delta")),
                        _sql_value(
                            component.get("rebalance_distortion_contribution")
                        ),
                        (
                            int(bool(component.get("rebalance_membership")))
                            if not pd.isna(component.get("rebalance_membership"))
                            else None
                        ),
                        component.get("company_id"),
                        _sql_value(component.get("modified_cap_ratio")),
                        _sql_value(component.get("modified_market_cap_mass")),
                        component.get("rebalance_input_status"),
                    )
                )
            connection.executemany(
                """
                INSERT INTO snapshot_components (
                    snapshot_id, ticker, company_name, actual_weight, float_weight,
                    counterfactual_weight, weight_delta, weight_ratio,
                    distortion_contribution, price, float_shares, reference_shares,
                    security_type, reference_source, acwi_weight, acwi_listing,
                    data_status, rebalance_weight, rebalance_reference_weight,
                    rebalance_weight_change, rebalance_weight_delta,
                    rebalance_distortion_contribution, rebalance_membership,
                    rebalance_company_id, modified_cap_ratio,
                    modified_market_cap_mass, rebalance_input_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            connection.commit()
        return snapshot_id

    def get_current(
        self,
        universe: str | None = None,
        weighting_basis: str | None = None,
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            if universe is None and weighting_basis is None:
                row = connection.execute(
                    "SELECT * FROM snapshots ORDER BY timestamp DESC, snapshot_id DESC LIMIT 1"
                ).fetchone()
            elif universe is not None and weighting_basis is None:
                row = connection.execute(
                    "SELECT * FROM snapshots WHERE universe = ? "
                    "ORDER BY timestamp DESC, snapshot_id DESC LIMIT 1",
                    (universe,),
                ).fetchone()
            elif universe is None:
                row = connection.execute(
                    "SELECT * FROM snapshots WHERE weighting_basis = ? "
                    "ORDER BY timestamp DESC, snapshot_id DESC LIMIT 1",
                    (weighting_basis,),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM snapshots WHERE universe = ? AND weighting_basis = ? "
                    "ORDER BY timestamp DESC, snapshot_id DESC LIMIT 1",
                    (universe, weighting_basis),
                ).fetchone()
        return dict(row) if row else None

    def get_current_by_universe(
        self, weighting_basis: str = "float"
    ) -> dict[str, dict[str, Any]]:
        return {
            universe: snapshot
            for universe in ("non_ucits", "ucits")
            if (snapshot := self.get_current(universe, weighting_basis)) is not None
        }

    def get_history(
        self,
        limit: int = 365,
        universe: str | None = None,
        weighting_basis: str | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 5000))
        with self.connect() as connection:
            if universe is None and weighting_basis is None:
                rows = connection.execute(
                    "SELECT * FROM snapshots ORDER BY timestamp DESC, snapshot_id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            elif universe is not None and weighting_basis is None:
                rows = connection.execute(
                    "SELECT * FROM snapshots WHERE universe = ? "
                    "ORDER BY timestamp DESC, snapshot_id DESC LIMIT ?",
                    (universe, limit),
                ).fetchall()
            elif universe is None:
                rows = connection.execute(
                    "SELECT * FROM snapshots WHERE weighting_basis = ? "
                    "ORDER BY timestamp DESC, snapshot_id DESC LIMIT ?",
                    (weighting_basis, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM snapshots WHERE universe = ? AND weighting_basis = ? "
                    "ORDER BY timestamp DESC, snapshot_id DESC LIMIT ?",
                    (universe, weighting_basis, limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def get_components(
        self,
        snapshot_id: int | None = None,
        universe: str | None = None,
        weighting_basis: str | None = None,
    ) -> list[dict[str, Any]]:
        if snapshot_id is None:
            current = self.get_current(universe, weighting_basis)
            if current is None:
                return []
            snapshot_id = int(current["snapshot_id"])
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM snapshot_components
                WHERE snapshot_id = ?
                ORDER BY distortion_contribution DESC, ticker ASC
                """,
                (snapshot_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_snapshot(self, snapshot_id: int) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
            ).fetchone()
        return dict(row) if row else None


def _sql_value(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
