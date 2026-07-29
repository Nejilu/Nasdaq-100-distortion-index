"""Snapshot-scoped Streamlit caches.

Every persisted snapshot is immutable. Using its id as the cache key avoids
re-reading SQLite and rebuilding view models during widget-only reruns.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from database import SnapshotDatabase
from ndx_wdi.domain.rebalance_analytics import (
    AnnualRebalanceAnalysis,
    analyze_annual_rebalance,
)


@st.cache_resource(show_spinner=False)
def get_database(path: str) -> SnapshotDatabase:
    return SnapshotDatabase(path)


@st.cache_data(show_spinner=False, max_entries=32)
def load_components(path: str, snapshot_id: int) -> pd.DataFrame:
    return pd.DataFrame(
        get_database(path).get_components(snapshot_id)
    )


@st.cache_data(show_spinner=False, max_entries=24)
def load_active_share(
    path: str,
    snapshot_id: int,
) -> tuple[dict[str, object] | None, pd.DataFrame]:
    database = get_database(path)
    summary = database.get_active_share(snapshot_id)
    components = pd.DataFrame(
        database.get_active_share_components(snapshot_id)
    )
    return summary, components


@st.cache_data(show_spinner=False, max_entries=24)
def load_annual_analysis(
    path: str,
    snapshot_id: int,
) -> AnnualRebalanceAnalysis:
    components = load_components(path, snapshot_id)
    return analyze_annual_rebalance(components)


@st.cache_data(show_spinner=False, max_entries=8)
def load_quarterly_history(
    path: str,
    modified_at_ns: int | None,
) -> pd.DataFrame:
    del modified_at_ns
    history_path = Path(path)
    if not history_path.exists():
        return pd.DataFrame()
    history = pd.read_csv(history_path)
    required = {
        "report_date",
        "ndx_wdi",
        "ndx_wdi_raw",
        "coverage_ratio",
        "matched_count",
        "estimated_count",
        "excluded_non_comparable_count",
        "rebalance_type",
    }
    if required.difference(history.columns):
        return pd.DataFrame()
    history["report_date"] = pd.to_datetime(
        history["report_date"],
        errors="coerce",
    )
    return history.dropna(
        subset=["report_date", "ndx_wdi"]
    ).sort_values("report_date")
