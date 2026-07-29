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

    def get_annual_selection(self, holdings: pd.DataFrame):
        return fallback_current_selection(
            holdings,
            reason="Deterministic test composition.",
        )


class _LiveSpxHoldingsProvider:
    def __init__(self, universe: str) -> None:
        self.source_name = f"test_{universe}_spx_holdings"
        self.reference_fund = "IVV" if universe == "non_ucits" else "CSPX"
        self.holdings_as_of = "2026-07-24"
        self.failures: tuple[str, ...] = ()

    def get_holdings(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": ["A", "B", "D"],
                "company_name": ["Alpha", "Beta", "Delta"],
                "actual_weight": [0.20, 0.20, 0.60],
            }
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
    monkeypatch.setattr(
        snapshot_service,
        "build_spx_holdings_chain",
        lambda universe: _LiveSpxHoldingsProvider(universe),
    )


def test_live_snapshot_roundtrip(tmp_path, live_sources):
    path = tmp_path / "snapshots.sqlite3"
    outcome = recompute_snapshot(db_path=path)
    database = SnapshotDatabase(path)

    current = database.get_current()
    components = database.get_components()
    active_share = database.get_active_share()
    active_components = database.get_active_share_components()

    assert current is not None
    assert current["snapshot_id"] == outcome.snapshot_id
    assert current["universe"] == "non_ucits"
    assert current["status"] == "complete"
    assert current["holdings_source"] == "test_non_ucits_live_holdings"
    assert current["market_data_source"] == "test_acwi_reference"
    assert current["reference_data_as_of"] == "2026-07-24"
    assert current["rebalance_ndx_wdi"] is not None
    assert current["rebalance_method"] == "annual_modified_market_cap_2026"
    assert len(components) == 3
    assert {row["reference_source"] for row in components} == {"ishares_acwi"}
    assert all(row["rebalance_weight"] is not None for row in components)
    assert active_share is not None
    assert active_share["spx_reference_fund"] == "IVV"
    assert active_share["active_share"] > 0
    assert {row["ticker"] for row in active_components} == {"A", "B", "C", "D"}


def test_rebalance_failure_preserves_live_snapshot(
    tmp_path,
    monkeypatch,
    live_sources,
):
    path = tmp_path / "rebalance-failure.sqlite3"

    def fail_rebalance(*_args, **_kwargs):
        raise ValueError("No valid modified-cap inputs")

    monkeypatch.setattr(snapshot_service, "simulate_rebalance", fail_rebalance)

    outcome = recompute_snapshot(db_path=path)
    database = SnapshotDatabase(path)
    current = database.get_current()
    components = database.get_components()
    active_share = database.get_active_share()
    active_components = database.get_active_share_components()

    assert outcome.rebalance is None
    assert outcome.result.snapshot_status == "complete"
    assert current is not None
    assert current["status"] == "complete"
    assert current["rebalance_ndx_wdi"] is None
    assert current["rebalance_status"] is None
    assert (
        "Nasdaq annual reconstitution: ValueError: "
        "No valid modified-cap inputs"
    ) in current["source_failures"]
    assert all(row["rebalance_weight"] is None for row in components)
    assert active_share is not None
    assert active_share["rebalanced_active_share"] is None


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
    active_response = client.get(
        "/api/active-share?universe=non_ucits&ranking=ndx_overweights&limit=2"
    )
    assert active_response.status_code == 200
    active_payload = active_response.json()
    assert active_payload["summary"]["spx_reference_fund"] == "IVV"
    assert len(active_payload["components"]) == 2

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


def test_partial_wdi_coverage_does_not_change_published_active_share(
    tmp_path,
    monkeypatch,
    live_sources,
):
    class PartialAcwiProvider(_LiveAcwiProvider):
        def build_reference(self, holdings, market_data):
            result = super().build_reference(holdings, market_data)
            missing = result["ticker"].eq("C")
            result.loc[missing, "reference_weight_raw"] = None
            result.loc[missing, "reference_status"] = (
                "missing_float_yfinance_fallback"
            )
            return result

    class IdenticalSpxProvider(_LiveSpxHoldingsProvider):
        def get_holdings(self):
            return _LiveHoldingsProvider("non_ucits").get_holdings()

    monkeypatch.setattr(
        snapshot_service,
        "IsharesAcwiFloatWeightsProvider",
        PartialAcwiProvider,
    )
    monkeypatch.setattr(
        snapshot_service,
        "build_spx_holdings_chain",
        lambda universe: IdenticalSpxProvider(universe),
    )

    outcome = recompute_snapshot(db_path=tmp_path / "partial-active-share.sqlite3")

    assert outcome.result.snapshot_status == "partial_coverage"
    assert outcome.active_share is not None
    assert outcome.active_share.active_share == pytest.approx(0.0)


def test_global_yfinance_fallback_rejects_inconsistent_float(
    tmp_path,
    monkeypatch,
    live_sources,
):
    class FailingAcwiProvider(_LiveAcwiProvider):
        def build_reference(self, holdings, market_data):
            raise ValueError("ACWI unavailable")

    class InconsistentMarketDataProvider(_LiveMarketDataProvider):
        def get_market_data(self, tickers):
            result = super().get_market_data(tickers)
            bad = result["ticker"].eq("C")
            result.loc[bad, "float_shares"] = 1_000.0
            return result

    monkeypatch.setattr(
        snapshot_service,
        "IsharesAcwiFloatWeightsProvider",
        FailingAcwiProvider,
    )
    monkeypatch.setattr(
        snapshot_service,
        "YFinanceMarketDataProvider",
        InconsistentMarketDataProvider,
    )

    outcome = recompute_snapshot(db_path=tmp_path / "invalid-fallback.sqlite3")
    components = outcome.result.components.set_index("ticker")

    assert outcome.market_data_source.endswith("_global_fallback")
    assert outcome.result.snapshot_status == "degraded_partial_coverage"
    assert outcome.result.invalid_float_count == 1
    assert outcome.result.coverage_ratio == pytest.approx(0.8)
    assert components.loc["C", "data_status"] == "invalid_yfinance_fallback"
    assert pd.isna(components.loc["C", "counterfactual_weight"])


def test_complete_global_yfinance_fallback_is_marked_degraded(
    tmp_path,
    monkeypatch,
    live_sources,
):
    class FailingAcwiProvider(_LiveAcwiProvider):
        def build_reference(self, holdings, market_data):
            raise ValueError("ACWI unavailable")

    monkeypatch.setattr(
        snapshot_service,
        "IsharesAcwiFloatWeightsProvider",
        FailingAcwiProvider,
    )

    outcome = recompute_snapshot(db_path=tmp_path / "degraded-fallback.sqlite3")

    assert outcome.result.coverage_ratio == pytest.approx(1.0)
    assert outcome.result.snapshot_status == "degraded_fallback"


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
