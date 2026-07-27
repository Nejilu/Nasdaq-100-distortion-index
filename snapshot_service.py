"""Snapshot orchestration independent from the API and dashboard surfaces."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from acwi_weights_provider import (
    DEFAULT_ACWI_URL,
    IsharesAcwiFloatWeightsProvider,
    classify_security_types,
)
from database import SnapshotDatabase
from distortion_engine import DistortionResult, calculate_distortion
from market_data_provider import YFinanceMarketDataProvider
from nasdaq100_rebalance import (
    NasdaqPublicUniverseProvider,
    RebalanceResult,
    fallback_current_selection,
    simulate_rebalance,
)
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
    rebalance: RebalanceResult | None = None

    def summary(self) -> dict[str, object]:
        summary = {
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
        if self.rebalance is not None:
            summary.update(
                {
                    "rebalance_ndx_wdi": self.rebalance.ndx_wdi,
                    "rebalance_coverage_ratio": self.rebalance.coverage_ratio,
                    "rebalance_constituent_count": self.rebalance.constituent_count,
                    "rebalance_status": self.rebalance.status,
                    "rebalance_method": self.rebalance.method,
                    "rebalance_reference_date": self.rebalance.reference_date,
                    "rebalance_additions": list(self.rebalance.additions),
                    "rebalance_removals": list(self.rebalance.removals),
                    "rebalance_data_source": self.rebalance.data_source,
                    "rebalance_acwi_conversion_scale": (
                        self.rebalance.acwi_conversion_scale
                    ),
                    "rebalance_acwi_calibration_count": (
                        self.rebalance.acwi_calibration_count
                    ),
                    "rebalance_notes": list(self.rebalance.notes),
                }
            )
        return summary


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
    try:
        selection = NasdaqPublicUniverseProvider(
            timeout=int(os.getenv("HTTP_TIMEOUT_SECONDS", "30")),
            cache_path=os.getenv(
                "NASDAQ_UNIVERSE_CACHE_PATH",
                "data/nasdaq_public_universe_cache.csv",
            ),
        ).get_quarterly_selection(holdings)
    except Exception as exc:
        failure = f"Nasdaq rebalance universe: {type(exc).__name__}: {exc}"
        source_failures.append(failure)
        selection = fallback_current_selection(holdings, reason=failure)

    combined_holdings = _combined_rebalance_holdings(holdings, selection.securities)
    market_data = market_provider.get_market_data(combined_holdings["ticker"].tolist())
    reference_data_as_of = None
    rebalance_reference: pd.DataFrame
    if weighting_basis == "float":
        acwi_provider = IsharesAcwiFloatWeightsProvider(
            url=os.getenv("ACWI_HOLDINGS_URL", DEFAULT_ACWI_URL),
            timeout=int(os.getenv("HTTP_TIMEOUT_SECONDS", "30")),
        )
        try:
            references = acwi_provider.build_reference(combined_holdings, market_data)
        except Exception as exc:
            source_failures.append(
                f"{acwi_provider.source_name}: {type(exc).__name__}: {exc}"
            )
            market_data = market_data.merge(
                classify_security_types(combined_holdings),
                on="ticker",
                how="left",
            )
            market_data["reference_weight_raw"] = (
                market_data["price"] * market_data["float_shares"]
            )
            market_data["reference_status"] = np.where(
                market_data["reference_weight_raw"].gt(0),
                "valid_yfinance_fallback",
                "missing_float_yfinance_fallback",
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
        rebalance_reference = market_data.copy()
        rebalance_reference["modified_float_mass_raw"] = rebalance_reference[
            "reference_weight_raw"
        ]
        rebalance_reference["counterfactual_reference_raw"] = rebalance_reference[
            "reference_weight_raw"
        ]
    else:
        market_data = market_data.merge(
            classify_security_types(combined_holdings),
            on="ticker",
            how="left",
        )
        market_data["reference_source"] = "yfinance_total_shares"
        market_source = market_provider.source_name
        acwi_provider = IsharesAcwiFloatWeightsProvider(
            url=os.getenv("ACWI_HOLDINGS_URL", DEFAULT_ACWI_URL),
            timeout=int(os.getenv("HTTP_TIMEOUT_SECONDS", "30")),
        )
        try:
            float_references = acwi_provider.build_reference(
                combined_holdings, market_data
            )
            reference_data_as_of = acwi_provider.holdings_as_of
            rebalance_reference = market_data.merge(
                float_references,
                on="ticker",
                how="left",
                suffixes=("", "_float_reference"),
            )
        except Exception as exc:
            source_failures.append(
                f"{acwi_provider.source_name}: {type(exc).__name__}: {exc}"
            )
            rebalance_reference = market_data.copy()
            rebalance_reference["reference_weight_raw"] = (
                rebalance_reference["price"] * rebalance_reference["float_shares"]
            )
        rebalance_reference["modified_float_mass_raw"] = rebalance_reference[
            "reference_weight_raw"
        ]
        rebalance_reference["counterfactual_reference_raw"] = (
            rebalance_reference["price"] * rebalance_reference["shares_outstanding"]
        )
    result = calculate_distortion(
        holdings,
        market_data,
        coverage_threshold=coverage_threshold,
        weighting_basis=weighting_basis,
    )
    rebalance = simulate_rebalance(
        holdings,
        rebalance_reference,
        selection,
        rebalance_type="quarterly",
    )
    result = replace(
        result,
        components=_merge_rebalance_components(result.components, rebalance.components),
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
        rebalance=rebalance,
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
        rebalance=rebalance,
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


def _combined_rebalance_holdings(
    holdings: pd.DataFrame,
    selection_securities: pd.DataFrame,
) -> pd.DataFrame:
    """Append hypothetical additions so market/reference providers can price them."""
    current = holdings[["ticker", "company_name", "actual_weight"]].copy()
    selected = selection_securities.loc[
        selection_securities["selected"].fillna(False),
        ["ticker", "company_name"],
    ].copy()
    selected["actual_weight"] = 0.0
    return (
        pd.concat([current, selected], ignore_index=True)
        .drop_duplicates("ticker", keep="first")
        .reset_index(drop=True)
    )


def _merge_rebalance_components(
    current_components: pd.DataFrame,
    rebalance_components: pd.DataFrame,
) -> pd.DataFrame:
    rebalance_columns = [
        "ticker",
        "rebalance_weight",
        "rebalance_reference_weight",
        "rebalance_weight_change",
        "rebalance_weight_delta",
        "rebalance_distortion_contribution",
        "rebalance_membership",
        "company_id",
        "modified_cap_ratio",
        "modified_market_cap_mass",
        "rebalance_input_status",
    ]
    available = [
        column for column in rebalance_columns if column in rebalance_components
    ]
    result = current_components.merge(
        rebalance_components[available].drop_duplicates("ticker"),
        on="ticker",
        how="outer",
    )
    metadata = rebalance_components.set_index("ticker")
    for column in current_components.columns:
        if column not in result:
            result[column] = np.nan
    for column in ["company_name", "security_type", "reference_source"]:
        if column in metadata and column in result:
            result[column] = result[column].fillna(result["ticker"].map(metadata[column]))
    result["actual_weight"] = result["actual_weight"].fillna(0.0)
    result["data_status"] = result["data_status"].fillna("rebalance_addition")
    return result
