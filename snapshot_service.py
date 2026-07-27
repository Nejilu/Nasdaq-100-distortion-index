"""Snapshot orchestration independent from the API and dashboard surfaces."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from acwi_weights_provider import (
    DEFAULT_ACWI_URL,
    IsharesAcwiFloatWeightsProvider,
    classify_security_types,
)
from database import SnapshotDatabase
from distortion_engine import DistortionResult, calculate_distortion
from market_data_provider import YFinanceMarketDataProvider
from qqq_holdings_provider import (
    DEFAULT_CNDX_URL,
    DEFAULT_EQQQ_URL,
    DEFAULT_IQQ_URL,
    CsvHoldingsProvider,
    HoldingsProviderChain,
    HttpCsvHoldingsProvider,
    InvescoQQQHoldingsProvider,
    IsharesSpreadsheetXmlHoldingsProvider,
)


UNIVERSES = ("non_ucits", "ucits")


@dataclass(frozen=True)
class RecomputeOutcome:
    snapshot_id: int
    timestamp: str
    result: DistortionResult
    holdings_source: str
    market_data_source: str
    universe: str = "non_ucits"
    reference_fund: str | None = None
    holdings_as_of: str | None = None
    reference_data_as_of: str | None = None
    source_failures: tuple[str, ...] = ()

    def summary(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "ndx_wdi": self.result.ndx_wdi,
            "coverage_ratio": self.result.coverage_ratio,
            "constituent_count": self.result.constituent_count,
            "missing_float_count": self.result.missing_float_count,
            "invalid_float_count": self.result.invalid_float_count,
            "missing_reference_shares_count": self.result.missing_reference_shares_count,
            "missing_price_count": self.result.missing_price_count,
            "snapshot_status": self.result.snapshot_status,
            "weighting_basis": self.result.weighting_basis,
            "universe": self.universe,
            "reference_fund": self.reference_fund,
            "holdings_as_of": self.holdings_as_of,
            "reference_data_as_of": self.reference_data_as_of,
            "holdings_source": self.holdings_source,
            "market_data_source": self.market_data_source,
            "source_failures": list(self.source_failures),
        }


def recompute_snapshot(
    *,
    db_path: str | Path | None = None,
    holdings_csv: str | Path | None = None,
    universe: str = "non_ucits",
    weighting_basis: str = "float",
) -> RecomputeOutcome:
    """Compute and persist one snapshot from live holdings and market data."""
    if universe not in UNIVERSES:
        raise ValueError(f"universe must be one of: {UNIVERSES}.")
    if weighting_basis not in {"float", "total"}:
        raise ValueError("weighting_basis must be float or total.")
    database = SnapshotDatabase(db_path or os.getenv("NDX_DB_PATH", "data/ndx_wdi.sqlite3"))
    coverage_threshold = float(os.getenv("NDX_COVERAGE_THRESHOLD", "0.99"))

    holdings_provider = build_holdings_chain(universe, holdings_csv=holdings_csv)
    market_provider = YFinanceMarketDataProvider(
        max_workers=int(os.getenv("YFINANCE_MAX_WORKERS", "8")),
        cache_dir=os.getenv("YFINANCE_CACHE_DIR", "data/yfinance_cache"),
    )
    holdings = holdings_provider.get_holdings()
    source_failures = list(holdings_provider.failures)
    holdings_as_of = holdings_provider.holdings_as_of
    market_data = market_provider.get_market_data(holdings["ticker"].tolist())
    reference_data_as_of = None
    if weighting_basis == "float":
        acwi_provider = IsharesAcwiFloatWeightsProvider(
            url=os.getenv("ACWI_HOLDINGS_URL", DEFAULT_ACWI_URL),
            timeout=int(os.getenv("HTTP_TIMEOUT_SECONDS", "30")),
        )
        try:
            references = acwi_provider.build_reference(holdings, market_data)
        except Exception as exc:
            source_failures.append(
                f"{acwi_provider.source_name}: {type(exc).__name__}: {exc}"
            )
            market_data = market_data.merge(
                classify_security_types(holdings),
                on="ticker",
                how="left",
            )
            market_source = f"{market_provider.source_name}_global_fallback"
        else:
            market_data = market_data.merge(references, on="ticker", how="left")
            reference_data_as_of = acwi_provider.holdings_as_of
            reference_sources = set(
                references["reference_source"].dropna().astype(str)
            )
            market_source = acwi_provider.source_name
            if "yfinance_fallback" in reference_sources:
                market_source += "+yfinance_fallback"
    else:
        market_data = market_data.merge(
            classify_security_types(holdings),
            on="ticker",
            how="left",
        )
        market_data["reference_source"] = "yfinance_total_shares"
        market_source = market_provider.source_name
    result = calculate_distortion(
        holdings,
        market_data,
        coverage_threshold=coverage_threshold,
        weighting_basis=weighting_basis,
    )
    holdings_source = holdings_provider.source_name
    reference_fund = holdings_provider.reference_fund

    timestamp = datetime.now(timezone.utc).isoformat()
    snapshot_id = database.save_snapshot(
        result,
        timestamp=timestamp,
        universe=universe,
        reference_fund=reference_fund,
        holdings_as_of=holdings_as_of,
        reference_data_as_of=reference_data_as_of,
        source_failures=" | ".join(source_failures) or None,
        holdings_source=holdings_source,
        market_data_source=market_source,
        weighting_basis=weighting_basis,
    )
    return RecomputeOutcome(
        snapshot_id=snapshot_id,
        timestamp=timestamp,
        result=result,
        holdings_source=holdings_source,
        market_data_source=market_source,
        universe=universe,
        reference_fund=reference_fund,
        holdings_as_of=holdings_as_of,
        reference_data_as_of=reference_data_as_of,
        source_failures=tuple(source_failures),
    )


def recompute_all_snapshots(
    *,
    db_path: str | Path | None = None,
    weighting_basis: str = "float",
) -> list[RecomputeOutcome]:
    return [
        recompute_snapshot(
            db_path=db_path,
            universe=universe,
            weighting_basis=weighting_basis,
        )
        for universe in UNIVERSES
    ]


def build_holdings_chain(
    universe: str, *, holdings_csv: str | Path | None = None
) -> HoldingsProviderChain:
    timeout = int(os.getenv("HTTP_TIMEOUT_SECONDS", "30"))
    providers = []
    configured_csv = holdings_csv or os.getenv(
        "NON_UCITS_HOLDINGS_CSV" if universe == "non_ucits" else "UCITS_HOLDINGS_CSV"
    )
    if universe == "non_ucits" and not configured_csv:
        configured_csv = os.getenv("QQQ_HOLDINGS_CSV")
    if configured_csv:
        providers.append(
            CsvHoldingsProvider(
                configured_csv,
                source_name=f"configured_{universe}_csv",
                reference_fund="configured_csv",
            )
        )

    if universe == "non_ucits":
        providers.extend(
            [
                IsharesSpreadsheetXmlHoldingsProvider(
                    url=os.getenv("IQQ_HOLDINGS_URL", DEFAULT_IQQ_URL), timeout=timeout
                ),
                InvescoQQQHoldingsProvider(
                    url=os.getenv("QQQ_HOLDINGS_URL", InvescoQQQHoldingsProvider.url),
                    timeout=timeout,
                ),
            ]
        )
        extra_urls = os.getenv("NON_UCITS_FALLBACK_URLS", "")
    else:
        providers.extend(
            [
                HttpCsvHoldingsProvider(
                    url=os.getenv("CNDX_HOLDINGS_URL", DEFAULT_CNDX_URL),
                    source_name="ishares_cndx_public_holdings",
                    reference_fund="CNDX",
                    timeout=timeout,
                ),
                HttpCsvHoldingsProvider(
                    url=os.getenv("EQQQ_HOLDINGS_URL", DEFAULT_EQQQ_URL),
                    source_name="invesco_eqqq_public_holdings",
                    reference_fund="EQQQ",
                    timeout=timeout,
                ),
            ]
        )
        extra_urls = os.getenv("UCITS_FALLBACK_URLS", "")

    for index, url in enumerate(filter(None, (item.strip() for item in extra_urls.split(","))), 1):
        providers.append(
            HttpCsvHoldingsProvider(
                url=url,
                source_name=f"configured_{universe}_url_{index}",
                reference_fund="configured_url",
                timeout=timeout,
            )
        )
    return HoldingsProviderChain(providers)
