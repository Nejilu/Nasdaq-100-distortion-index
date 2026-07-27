import pandas as pd
import pytest
from fastapi.testclient import TestClient

import snapshot_service
from api import _database_for_path, app, get_database
from database import SnapshotDatabase
from nasdaq100_rebalance import fallback_current_selection
from snapshot_service import recompute_all_snapshots, recompute_snapshot


class _LiveHoldingsProvider:
    def __init__(self, universe: str) -> None:
        self.universe = universe
        self.source_name = f"test_{universe}_live_holdings"
        self.reference_fund = "TEST_QQQ" if universe == "non_ucits" else "TEST_CNDX"
        self.holdings_as_of = "2026-07-25"
        self.failures: tuple[str, ...] = ()

    def get_holdings(self) -> pd.DataFrame:
        weights = (
            [0.50, 0.30, 0.20]
            if self.universe == "non_ucits"
            else [0.45, 0.35, 0.20]
        )
        return pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "company_name": ["Alpha", "Beta", "Gamma"],
                "actual_weight": weights,
            }
        )


class _LiveMarketDataProvider:
    source_name = "test_live_market_data"

    def __init__(self, **_: object) -> None:
        pass

    def get_market_data(self, tickers: list[str]) -> pd.DataFrame:
        rows = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "price": [1.0, 1.0, 1.0],
                "float_shares": [60.0, 25.0, 15.0],
                "shares_outstanding": [70.0, 30.0, 20.0],
                "market_cap": [70.0, 30.0, 20.0],
            }
        )
        return rows.loc[rows["ticker"].isin(tickers)].reset_index(drop=True)


class _LiveAcwiProvider:
    source_name = "test_acwi_reference"
    holdings_as_of = "2026-07-24"

    def __init__(self, **_: object) -> None:
        pass

    def build_reference(
        self, holdings: pd.DataFrame, market_data: pd.DataFrame
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "reference_weight_raw": [60.0, 25.0, 15.0],
                "reference_source": ["ishares_acwi"] * 3,
                "security_type": ["Ordinary share"] * 3,
                "acwi_weight": [0.60, 0.25, 0.15],
                "acwi_market_value": [60.0, 25.0, 15.0],
                "acwi_listing": ["United States / NASDAQ"] * 3,
                "reference_status": ["valid_acwi"] * 3,
            }
        )


class _LiveNasdaqUniverseProvider:
    def __init__(self, **_: object) -> None:
        pass

    def get_quarterly_selection(self, holdings: pd.DataFrame):
        return fallback_current_selection(
            holdings,
            reason="Deterministic test composition.",
        )


@pytest.fixture
def live_sources(monkeypatch):
    monkeypatch.setattr(
        snapshot_service,
        "build_holdings_chain",
        lambda universe, holdings_csv=None: _LiveHoldingsProvider(universe),
    )
    monkeypatch.setattr(
        snapshot_service, "YFinanceMarketDataProvider", _LiveMarketDataProvider
    )
    monkeypatch.setattr(
        snapshot_service, "IsharesAcwiFloatWeightsProvider", _LiveAcwiProvider
    )
    monkeypatch.setattr(
        snapshot_service, "NasdaqPublicUniverseProvider", _LiveNasdaqUniverseProvider
    )


def test_live_snapshot_roundtrip(tmp_path, live_sources):
    path = tmp_path / "snapshots.sqlite3"
    outcome = recompute_snapshot(db_path=path)
    database = SnapshotDatabase(path)

    current = database.get_current()
    components = database.get_components()

    assert current is not None
    assert current["snapshot_id"] == outcome.snapshot_id
    assert current["universe"] == "non_ucits"
    assert current["status"] == "complete"
    assert current["holdings_source"] == "test_non_ucits_live_holdings"
    assert current["market_data_source"] == "test_acwi_reference"
    assert current["reference_data_as_of"] == "2026-07-24"
    assert current["rebalance_ndx_wdi"] is not None
    assert current["rebalance_method"] == "quarterly_modified_market_cap_2026"
    assert len(components) == 3
    assert {row["reference_source"] for row in components} == {"ishares_acwi"}
    assert all(row["rebalance_weight"] is not None for row in components)


def test_minimal_api(tmp_path, monkeypatch, live_sources):
    path = tmp_path / "api.sqlite3"
    monkeypatch.setenv("NDX_DB_PATH", str(path))
    client = TestClient(app)

    response = client.post("/api/recompute", json={"universe": "all"})
    assert response.status_code == 201
    payload = response.json()["snapshots"]
    assert set(payload) == {"non_ucits", "ucits"}
    assert payload["ucits"]["snapshot_status"] == "complete"

    current_response = client.get("/api/current")
    assert current_response.status_code == 200
    assert set(current_response.json()["snapshots"]) == {"non_ucits", "ucits"}
    assert client.get("/api/current?universe=ucits").json()["reference_fund"] == "TEST_CNDX"
    assert len(client.get("/api/history").json()) == 2
    components = client.get("/api/components?universe=ucits").json()
    assert len(components) == 3
    contributors = client.get(
        "/api/components?universe=non_ucits&ranking=contributors&limit=2"
    ).json()
    assert len(contributors) == 2
    assert contributors[0]["distortion_contribution"] >= contributors[1][
        "distortion_contribution"
    ]

    total_response = client.post(
        "/api/recompute",
        json={"universe": "all", "weighting_basis": "total"},
    )
    assert total_response.status_code == 201
    total_current = client.get(
        "/api/current?universe=ucits&weighting_basis=total"
    ).json()
    assert total_current["weighting_basis"] == "total"
    total_components = client.get(
        "/api/components?universe=ucits&weighting_basis=total"
    ).json()
    assert all(row["counterfactual_weight"] is not None for row in total_components)


def test_api_rejects_removed_mode_field(tmp_path, monkeypatch):
    monkeypatch.setenv("NDX_DB_PATH", str(tmp_path / "api.sqlite3"))
    client = TestClient(app)

    response = client.post("/api/recompute", json={"mode": "live"})

    assert response.status_code == 422


def test_both_universes_are_persisted_separately(tmp_path, live_sources):
    path = tmp_path / "both.sqlite3"
    outcomes = recompute_all_snapshots(db_path=path)
    database = SnapshotDatabase(path)

    assert {outcome.universe for outcome in outcomes} == {"non_ucits", "ucits"}
    current = database.get_current_by_universe()
    assert current["non_ucits"]["reference_fund"] == "TEST_QQQ"
    assert current["ucits"]["reference_fund"] == "TEST_CNDX"
    assert current["non_ucits"]["ndx_wdi"] != current["ucits"]["ndx_wdi"]


def test_total_basis_is_persisted_and_queried_separately(tmp_path, live_sources):
    path = tmp_path / "basis.sqlite3"
    recompute_all_snapshots(db_path=path, weighting_basis="float")
    total_outcomes = recompute_all_snapshots(db_path=path, weighting_basis="total")
    database = SnapshotDatabase(path)

    total = database.get_current_by_universe("total")
    floating = database.get_current_by_universe("float")
    components = database.get_components(universe="ucits", weighting_basis="total")

    assert {outcome.result.weighting_basis for outcome in total_outcomes} == {"total"}
    assert total["ucits"]["weighting_basis"] == "total"
    assert floating["ucits"]["weighting_basis"] == "float"
    assert total["ucits"]["snapshot_id"] != floating["ucits"]["snapshot_id"]
    assert all(row["counterfactual_weight"] is not None for row in components)
    assert all(row["float_weight"] is None for row in components)


def test_api_database_is_cached_per_configured_path(tmp_path, monkeypatch):
    first_path = tmp_path / "first.sqlite3"
    second_path = tmp_path / "second.sqlite3"
    _database_for_path.cache_clear()
    try:
        monkeypatch.setenv("NDX_DB_PATH", str(first_path))
        first = get_database()
        assert get_database() is first

        monkeypatch.setenv("NDX_DB_PATH", str(second_path))
        second = get_database()
        assert second is not first
        assert get_database() is second
    finally:
        _database_for_path.cache_clear()
