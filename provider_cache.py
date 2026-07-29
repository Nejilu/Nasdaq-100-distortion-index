"""Persistent provider cache shared by Streamlit, FastAPI, and workers."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd

from ndx_wdi.domain.rebalance import SelectionResult


CACHE_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS market_data_cache (
    ticker TEXT PRIMARY KEY,
    price REAL,
    float_shares REAL,
    shares_outstanding REAL,
    market_cap REAL,
    float_shares_status TEXT,
    shares_outstanding_status TEXT,
    market_data_error TEXT,
    price_fetched_at REAL,
    fundamentals_fetched_at REAL,
    fundamentals_attempted_at REAL
);

CREATE TABLE IF NOT EXISTS nasdaq_selection_cache (
    universe TEXT PRIMARY KEY,
    holdings_fingerprint TEXT NOT NULL,
    fetched_at REAL NOT NULL,
    selection_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS provider_frame_cache (
    cache_key TEXT PRIMARY KEY,
    fetched_at REAL NOT NULL,
    frame_json TEXT NOT NULL,
    attrs_json TEXT NOT NULL,
    source_name TEXT,
    reference_fund TEXT,
    holdings_as_of TEXT,
    failures_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS background_refresh_jobs (
    job_key TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    started_at REAL,
    finished_at REAL,
    error TEXT,
    snapshot_id INTEGER
);
"""


@dataclass(frozen=True)
class CachedSelection:
    selection: SelectionResult
    age_seconds: float
    holdings_match: bool
    is_fresh: bool


@dataclass(frozen=True)
class CachedProviderFrame:
    frame: pd.DataFrame
    age_seconds: float
    is_fresh: bool
    source_name: str | None
    reference_fund: str | None
    holdings_as_of: str | None
    failures: tuple[str, ...]


class ProviderCache:
    def __init__(
        self,
        path: str | Path = "data/provider_cache.sqlite3",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(CACHE_SCHEMA)
            connection.commit()

    def get_market_data(self, tickers: Sequence[str]) -> pd.DataFrame:
        normalized = list(
            dict.fromkeys(str(ticker).upper().strip() for ticker in tickers)
        )
        if not normalized:
            return pd.DataFrame()
        placeholders = ",".join("?" for _ in normalized)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM market_data_cache
                WHERE ticker IN ({placeholders})
                """,
                normalized,
            ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def upsert_market_data(self, rows: pd.DataFrame) -> None:
        if rows.empty:
            return
        columns = [
            "ticker",
            "price",
            "float_shares",
            "shares_outstanding",
            "market_cap",
            "float_shares_status",
            "shares_outstanding_status",
            "market_data_error",
            "price_fetched_at",
            "fundamentals_fetched_at",
            "fundamentals_attempted_at",
        ]
        records = []
        for row in rows.to_dict(orient="records"):
            records.append(
                tuple(_sqlite_value(row.get(column)) for column in columns)
            )
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO market_data_cache (
                    ticker, price, float_shares, shares_outstanding, market_cap,
                    float_shares_status, shares_outstanding_status,
                    market_data_error, price_fetched_at,
                    fundamentals_fetched_at, fundamentals_attempted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    price = excluded.price,
                    float_shares = excluded.float_shares,
                    shares_outstanding = excluded.shares_outstanding,
                    market_cap = excluded.market_cap,
                    float_shares_status = excluded.float_shares_status,
                    shares_outstanding_status = excluded.shares_outstanding_status,
                    market_data_error = excluded.market_data_error,
                    price_fetched_at = excluded.price_fetched_at,
                    fundamentals_fetched_at = excluded.fundamentals_fetched_at,
                    fundamentals_attempted_at = excluded.fundamentals_attempted_at
                """,
                records,
            )
            connection.commit()

    def get_selection(
        self,
        universe: str,
        holdings_fingerprint: str,
        *,
        max_age_seconds: float,
    ) -> CachedSelection | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT holdings_fingerprint, fetched_at, selection_json
                FROM nasdaq_selection_cache
                WHERE universe = ?
                """,
                (universe,),
            ).fetchone()
        if row is None:
            return None
        selection = _deserialize_selection(str(row["selection_json"]))
        age_seconds = max(0.0, time.time() - float(row["fetched_at"]))
        return CachedSelection(
            selection=selection,
            age_seconds=age_seconds,
            holdings_match=str(row["holdings_fingerprint"])
            == holdings_fingerprint,
            is_fresh=age_seconds <= max_age_seconds,
        )

    def save_selection(
        self,
        universe: str,
        holdings_fingerprint: str,
        selection: SelectionResult,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO nasdaq_selection_cache (
                    universe, holdings_fingerprint, fetched_at, selection_json
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(universe) DO UPDATE SET
                    holdings_fingerprint = excluded.holdings_fingerprint,
                    fetched_at = excluded.fetched_at,
                    selection_json = excluded.selection_json
                """,
                (
                    universe,
                    holdings_fingerprint,
                    time.time(),
                    _serialize_selection(selection),
                ),
            )
            connection.commit()

    def get_provider_frame(
        self,
        cache_key: str,
        *,
        max_age_seconds: float,
    ) -> CachedProviderFrame | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM provider_frame_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        frame = pd.read_json(
            StringIO(str(row["frame_json"])),
            orient="split",
        )
        frame.attrs.update(json.loads(str(row["attrs_json"])))
        age_seconds = max(0.0, time.time() - float(row["fetched_at"]))
        return CachedProviderFrame(
            frame=frame,
            age_seconds=age_seconds,
            is_fresh=age_seconds <= max_age_seconds,
            source_name=row["source_name"],
            reference_fund=row["reference_fund"],
            holdings_as_of=row["holdings_as_of"],
            failures=tuple(json.loads(str(row["failures_json"]))),
        )

    def save_provider_frame(
        self,
        cache_key: str,
        frame: pd.DataFrame,
        *,
        source_name: str | None = None,
        reference_fund: str | None = None,
        holdings_as_of: str | None = None,
        failures: Sequence[str] = (),
    ) -> None:
        frame_json = frame.to_json(
            orient="split",
            double_precision=15,
        )
        attrs_json = json.dumps(
            frame.attrs,
            ensure_ascii=True,
            separators=(",", ":"),
            default=_json_default,
        )
        failures_json = json.dumps(
            list(failures),
            ensure_ascii=True,
            separators=(",", ":"),
        )
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO provider_frame_cache (
                    cache_key, fetched_at, frame_json, attrs_json,
                    source_name, reference_fund, holdings_as_of, failures_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    fetched_at = excluded.fetched_at,
                    frame_json = excluded.frame_json,
                    attrs_json = excluded.attrs_json,
                    source_name = excluded.source_name,
                    reference_fund = excluded.reference_fund,
                    holdings_as_of = excluded.holdings_as_of,
                    failures_json = excluded.failures_json
                """,
                (
                    cache_key,
                    time.time(),
                    frame_json,
                    attrs_json,
                    source_name,
                    reference_fund,
                    holdings_as_of,
                    failures_json,
                ),
            )
            connection.commit()

    def set_job_status(
        self,
        job_key: str,
        status: str,
        *,
        error: str | None = None,
        snapshot_id: int | None = None,
    ) -> None:
        now = time.time()
        started_at = now if status == "running" else None
        finished_at = now if status in {"complete", "failed"} else None
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO background_refresh_jobs (
                    job_key, status, started_at, finished_at, error, snapshot_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_key) DO UPDATE SET
                    status = excluded.status,
                    started_at = CASE
                        WHEN excluded.status = 'running' THEN excluded.started_at
                        ELSE background_refresh_jobs.started_at
                    END,
                    finished_at = excluded.finished_at,
                    error = excluded.error,
                    snapshot_id = excluded.snapshot_id
                """,
                (
                    job_key,
                    status,
                    started_at,
                    finished_at,
                    error,
                    snapshot_id,
                ),
            )
            connection.commit()

    def get_job(self, job_key: str) -> dict[str, object] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM background_refresh_jobs
                WHERE job_key = ?
                """,
                (job_key,),
            ).fetchone()
        return dict(row) if row else None


def holdings_fingerprint(holdings: pd.DataFrame) -> str:
    tickers = sorted(
        {
            str(ticker).upper().strip()
            for ticker in holdings["ticker"].dropna()
            if str(ticker).strip()
        }
    )
    return hashlib.sha256("\n".join(tickers).encode("utf-8")).hexdigest()


def background_job_key(universe: str, weighting_basis: str) -> str:
    return f"nasdaq-selection:{universe}:{weighting_basis}"


def _serialize_selection(selection: SelectionResult) -> str:
    payload = {
        "securities": json.loads(
            selection.securities.to_json(orient="records")
        ),
        "selected_tickers": list(selection.selected_tickers),
        "selected_company_ids": list(selection.selected_company_ids),
        "additions": list(selection.additions),
        "removals": list(selection.removals),
        "status": selection.status,
        "source": selection.source,
        "as_of": selection.as_of,
        "eligible_company_count": selection.eligible_company_count,
        "notes": list(selection.notes),
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def _deserialize_selection(raw: str) -> SelectionResult:
    payload = json.loads(raw)
    return SelectionResult(
        securities=pd.DataFrame(payload["securities"]),
        selected_tickers=tuple(payload["selected_tickers"]),
        selected_company_ids=tuple(payload["selected_company_ids"]),
        additions=tuple(payload["additions"]),
        removals=tuple(payload["removals"]),
        status=str(payload["status"]),
        source=str(payload["source"]),
        as_of=str(payload["as_of"]),
        eligible_company_count=int(payload["eligible_company_count"]),
        notes=tuple(payload.get("notes", [])),
    )


def _sqlite_value(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _json_default(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    return str(value)
