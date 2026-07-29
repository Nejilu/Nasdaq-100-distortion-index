import pandas as pd

from ndx_wdi.ui import runtime


class _FakeDatabase:
    component_reads = 0

    def __init__(self, path: str) -> None:
        self.path = path

    def get_components(self, snapshot_id: int):
        type(self).component_reads += 1
        return [
            {
                "ticker": f"T{snapshot_id}",
                "actual_weight": 1.0,
            }
        ]


def test_component_cache_is_scoped_by_immutable_snapshot_id(monkeypatch):
    runtime.load_components.clear()
    runtime.get_database.clear()
    _FakeDatabase.component_reads = 0
    monkeypatch.setattr(runtime, "SnapshotDatabase", _FakeDatabase)

    first = runtime.load_components("test.sqlite3", 10)
    repeated = runtime.load_components("test.sqlite3", 10)
    next_snapshot = runtime.load_components("test.sqlite3", 11)

    assert _FakeDatabase.component_reads == 2
    pd.testing.assert_frame_equal(first, repeated)
    assert first.iloc[0]["ticker"] == "T10"
    assert next_snapshot.iloc[0]["ticker"] == "T11"


def test_quarterly_history_cache_invalidates_with_file_timestamp(tmp_path):
    runtime.load_quarterly_history.clear()
    path = tmp_path / "history.csv"
    columns = {
        "report_date": ["2026-03-31"],
        "ndx_wdi": [20.0],
        "ndx_wdi_raw": [19.0],
        "coverage_ratio": [0.9],
        "matched_count": [90],
        "estimated_count": [10],
        "excluded_non_comparable_count": [2],
        "rebalance_type": ["quarterly"],
    }
    pd.DataFrame(columns).to_csv(path, index=False)

    first = runtime.load_quarterly_history(
        str(path),
        path.stat().st_mtime_ns,
    )
    columns["ndx_wdi"] = [21.0]
    pd.DataFrame(columns).to_csv(path, index=False)
    second = runtime.load_quarterly_history(
        str(path),
        path.stat().st_mtime_ns,
    )

    assert first.iloc[0]["ndx_wdi"] == 20.0
    assert second.iloc[0]["ndx_wdi"] == 21.0
