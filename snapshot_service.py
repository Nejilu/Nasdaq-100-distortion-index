"""Snapshot orchestration independent from the API and dashboard surfaces."""

from __future__ import annotations

import logging
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
from background_jobs import submit_unique
from cached_providers import (
    CachedAcwiFloatWeightsProvider,
    CachedHoldingsProvider,
    provider_cache_key,
)
from database import SnapshotDatabase
from market_data_provider import (
    FLOAT_SHARES_OVERRIDE_STATUS,
    YFinanceMarketDataProvider,
)
from ndx_wdi.domain.active_share import ActiveShareResult, calculate_active_share
from ndx_wdi.domain.distortion import DistortionResult, calculate_distortion
from ndx_wdi.domain.market_quality import evaluate_float_observations
from ndx_wdi.domain.rebalance import (
    RebalanceResult,
    fallback_current_selection,
    simulate_rebalance,
)
from nasdaq100_rebalance import (
    NasdaqPublicUniverseProvider,
)
from observability import PipelineMetrics, performance_status, structured_event
from provider_cache import (
    ProviderCache,
    background_job_key,
    holdings_fingerprint,
)
from qqq_holdings_provider import (
    DEFAULT_CNDX_URL,
    DEFAULT_CSPX_URL,
    DEFAULT_EQQQ_URL,
    DEFAULT_IQQ_URL,
    DEFAULT_IVV_URL,
    CsvHoldingsProvider,
    HoldingsProviderChain,
    HttpCsvHoldingsProvider,
    InvescoQQQHoldingsProvider,
    IsharesSpreadsheetXmlHoldingsProvider,
)


UNIVERSES = ("non_ucits", "ucits")
LOGGER = logging.getLogger(__name__)


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
    active_share: ActiveShareResult | None = None
    performance_status: str = "unknown"
    timings_ms: dict[str, float] | None = None
    cache_statuses: dict[str, str] | None = None
    background_refresh_queued: bool = False

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
            "performance_status": self.performance_status,
            "timings_ms": dict(self.timings_ms or {}),
            "cache_statuses": dict(self.cache_statuses or {}),
            "background_refresh_queued": self.background_refresh_queued,
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
        if self.active_share is not None:
            summary.update(
                {
                    "active_share": self.active_share.active_share,
                    "rebalanced_active_share": (
                        self.active_share.rebalanced_active_share
                    ),
                    "spx_reference_fund": self.active_share.spx_reference_fund,
                    "spx_holdings_source": self.active_share.spx_holdings_source,
                    "spx_holdings_as_of": self.active_share.spx_holdings_as_of,
                    "active_share_status": self.active_share.status,
                }
            )
        return summary


def recompute_snapshot(
    *,
    db_path: str | Path | None = None,
    holdings_csv: str | Path | None = None,
    universe: str = "non_ucits",
    weighting_basis: str = "float",
    background_refresh: bool = True,
) -> RecomputeOutcome:
    """Compute and persist one snapshot from live holdings and market data."""
    metrics = PipelineMetrics()
    try:
        outcome = _recompute_snapshot(
            db_path=db_path,
            holdings_csv=holdings_csv,
            universe=universe,
            weighting_basis=weighting_basis,
            metrics=metrics,
            background_refresh=background_refresh,
        )
    except Exception as exc:
        metrics.finish()
        LOGGER.error(
            structured_event(
                "snapshot_recompute_failed",
                universe=universe,
                weighting_basis=weighting_basis,
                error_type=type(exc).__name__,
                error=str(exc),
                timings_ms=metrics.timings_ms,
                cache_statuses=metrics.cache_statuses,
            )
        )
        raise
    LOGGER.info(
        structured_event(
            "snapshot_recompute_complete",
            snapshot_id=outcome.snapshot_id,
            universe=universe,
            weighting_basis=weighting_basis,
            performance_status=outcome.performance_status,
            timings_ms=outcome.timings_ms,
            cache_statuses=outcome.cache_statuses,
        )
    )
    return outcome


def _recompute_snapshot(
    *,
    db_path: str | Path | None,
    holdings_csv: str | Path | None,
    universe: str,
    weighting_basis: str,
    metrics: PipelineMetrics,
    background_refresh: bool,
) -> RecomputeOutcome:
    if universe not in UNIVERSES:
        raise ValueError(f"universe must be one of: {UNIVERSES}.")
    if weighting_basis not in {"float", "total"}:
        raise ValueError("weighting_basis must be float or total.")
    database = SnapshotDatabase(db_path or os.getenv("NDX_DB_PATH", "data/ndx_wdi.sqlite3"))
    metrics.checkpoint("database_initialize")
    coverage_threshold = float(os.getenv("NDX_COVERAGE_THRESHOLD", "0.99"))
    provider_cache_path = os.getenv(
        "NDX_PROVIDER_CACHE_PATH",
        "data/provider_cache.sqlite3",
    )
    provider_cache = ProviderCache(provider_cache_path)
    provider_holdings_ttl_seconds = float(
        os.getenv("PROVIDER_HOLDINGS_TTL_SECONDS", "21600")
    )

    base_holdings_provider = build_holdings_chain(
        universe,
        holdings_csv=holdings_csv,
    )
    if holdings_csv is None:
        holdings_provider = CachedHoldingsProvider(
            provider=base_holdings_provider,
            cache=provider_cache,
            cache_key=provider_cache_key(
                f"ndx-holdings:{universe}",
                base_holdings_provider,
            ),
            ttl_seconds=provider_holdings_ttl_seconds,
        )
    else:
        holdings_provider = base_holdings_provider
    market_provider = YFinanceMarketDataProvider(
        max_workers=int(os.getenv("YFINANCE_MAX_WORKERS", "8")),
        batch_timeout_seconds=float(
            os.getenv("YFINANCE_BATCH_TIMEOUT_SECONDS", "60")
        ),
        cache_dir=os.getenv("YFINANCE_CACHE_DIR", "data/yfinance_cache"),
        persistent_cache_path=os.getenv(
            "YFINANCE_MARKET_CACHE_PATH",
            provider_cache_path,
        ),
        price_ttl_seconds=float(
            os.getenv("YFINANCE_PRICE_TTL_SECONDS", "600")
        ),
        fundamentals_ttl_seconds=float(
            os.getenv("YFINANCE_FUNDAMENTALS_TTL_SECONDS", "86400")
        ),
        failure_retry_seconds=float(
            os.getenv("YFINANCE_FAILURE_RETRY_SECONDS", "60")
        ),
    )
    holdings = holdings_provider.get_holdings()
    metrics.record_cache(
        "ndx_holdings",
        getattr(holdings_provider, "cache_status", "local_source_bypass"),
    )
    metrics.checkpoint("ndx_holdings")
    source_failures = list(holdings_provider.failures)
    holdings_as_of = holdings_provider.holdings_as_of
    current_holdings_fingerprint = holdings_fingerprint(holdings)
    selection_ttl_seconds = float(
        os.getenv("NASDAQ_SELECTION_TTL_SECONDS", "86400")
    )
    cached_selection = provider_cache.get_selection(
        universe,
        current_holdings_fingerprint,
        max_age_seconds=selection_ttl_seconds,
    )
    selection_refresh_needed = True
    if cached_selection is not None and cached_selection.holdings_match:
        selection = cached_selection.selection
        selection_cache_status = (
            "fresh_hit" if cached_selection.is_fresh else "stale_hit"
        )
        selection_refresh_needed = not cached_selection.is_fresh
    else:
        reason = (
            "Cached annual selection does not match current holdings; "
            "background refresh queued."
            if cached_selection is not None
            else "No cached annual selection; background refresh queued."
        )
        selection = fallback_current_selection(holdings, reason=reason)
        selection_cache_status = (
            "holdings_mismatch_current_fallback"
            if cached_selection is not None
            else "miss_current_fallback"
        )
    metrics.record_cache(
        "nasdaq_selection",
        selection_cache_status,
    )
    metrics.checkpoint("nasdaq_selection")

    combined_holdings = _combined_rebalance_holdings(holdings, selection.securities)
    market_data = market_provider.get_market_data(combined_holdings["ticker"].tolist())
    metrics.record_cache(
        "yfinance_market_data",
        getattr(market_provider, "cache_status", "not_observable"),
    )
    metrics.checkpoint("market_data")
    reference_data_as_of = None
    rebalance_reference: pd.DataFrame
    global_yfinance_fallback = False
    base_acwi_provider = IsharesAcwiFloatWeightsProvider(
        url=os.getenv("ACWI_HOLDINGS_URL", DEFAULT_ACWI_URL),
        timeout=int(os.getenv("HTTP_TIMEOUT_SECONDS", "30")),
    )
    acwi_provider = CachedAcwiFloatWeightsProvider(
        provider=base_acwi_provider,
        cache=provider_cache,
        cache_key=provider_cache_key("acwi-holdings", base_acwi_provider),
        ttl_seconds=provider_holdings_ttl_seconds,
    )
    if weighting_basis == "float":
        try:
            references = acwi_provider.build_reference(combined_holdings, market_data)
        except Exception as exc:
            global_yfinance_fallback = True
            source_failures.append(
                f"{acwi_provider.source_name}: {type(exc).__name__}: {exc}"
            )
            market_data = market_data.merge(
                classify_security_types(combined_holdings),
                on="ticker",
                how="left",
            )
            float_quality = evaluate_float_observations(market_data)
            market_data["reference_weight_raw"] = float_quality[
                "float_cap"
            ].where(float_quality["valid"])
            market_data["reference_source"] = "yfinance_fallback"
            market_data["reference_status"] = np.select(
                [
                    float_quality["valid"],
                    float_quality["inconsistent"],
                    ~float_quality["float_valid"],
                    ~float_quality["price_valid"],
                ],
                [
                    "valid_yfinance_fallback",
                    "invalid_yfinance_fallback",
                    "missing_float_yfinance_fallback",
                    "missing_price_yfinance_fallback",
                ],
                default="invalid_yfinance_fallback",
            )
            hardcoded_override = (
                market_data.get(
                    "float_shares_status",
                    pd.Series(None, index=market_data.index),
                )
                .astype("string")
                .eq(FLOAT_SHARES_OVERRIDE_STATUS)
                & float_quality["valid"]
            )
            market_data.loc[
                hardcoded_override, "reference_source"
            ] = "hardcoded_float_override"
            market_data.loc[
                hardcoded_override, "reference_status"
            ] = "valid_hardcoded_float_override"
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
            if "hardcoded_float_override" in reference_sources:
                market_source += "+hardcoded_float_override"
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
    metrics.record_cache(
        "acwi_holdings",
        acwi_provider.cache_status,
    )
    metrics.checkpoint("reference_data")
    result = calculate_distortion(
        holdings,
        market_data,
        coverage_threshold=coverage_threshold,
        weighting_basis=weighting_basis,
    )
    if global_yfinance_fallback:
        fallback_status = (
            "degraded_fallback"
            if result.coverage_ratio >= coverage_threshold
            else "degraded_partial_coverage"
        )
        result = replace(result, snapshot_status=fallback_status)
    metrics.checkpoint("distortion_calculation")
    rebalance: RebalanceResult | None = None
    try:
        rebalance = simulate_rebalance(
            holdings,
            rebalance_reference,
            selection,
            rebalance_type="annual",
        )
    except Exception as exc:
        source_failures.append(
            f"Nasdaq annual reconstitution: {type(exc).__name__}: {exc}"
        )
    else:
        result = replace(
            result,
            components=_merge_rebalance_components(
                result.components,
                rebalance.components,
            ),
        )
    metrics.checkpoint("annual_reconstitution")
    active_share: ActiveShareResult | None = None
    base_spx_provider = build_spx_holdings_chain(universe)
    spx_provider = CachedHoldingsProvider(
        provider=base_spx_provider,
        cache=provider_cache,
        cache_key=provider_cache_key(
            f"spx-holdings:{universe}",
            base_spx_provider,
        ),
        ttl_seconds=provider_holdings_ttl_seconds,
    )
    try:
        spx_holdings = spx_provider.get_holdings()
        source_failures.extend(spx_provider.failures)
        metrics.checkpoint("spx_holdings")
        active_share = calculate_active_share(
            _build_active_share_ndx_holdings(
                holdings,
                rebalance.components if rebalance is not None else None,
            ),
            spx_holdings,
            spx_reference_fund=spx_provider.reference_fund,
            spx_holdings_source=spx_provider.source_name,
            spx_holdings_as_of=spx_provider.holdings_as_of,
        )
    except Exception as exc:
        source_failures.append(
            f"S&P 500 holdings comparison: {type(exc).__name__}: {exc}"
        )
    metrics.record_cache(
        "spx_holdings",
        spx_provider.cache_status,
    )
    metrics.checkpoint("active_share_calculation")
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
        active_share=active_share,
    )
    metrics.checkpoint("database_persist")
    total_ms = metrics.finish()
    refresh_warn_seconds = float(
        os.getenv("NDX_REFRESH_WARN_SECONDS", "180")
    )
    refresh_performance_status = performance_status(
        total_ms,
        refresh_warn_seconds,
    )
    background_refresh_queued = False
    if (
        selection_refresh_needed
        and background_refresh
        and _background_refresh_enabled()
    ):
        background_refresh_queued = _schedule_nasdaq_refresh(
            db_path=database.path,
            holdings=holdings,
            holdings_csv=holdings_csv,
            universe=universe,
            weighting_basis=weighting_basis,
            provider_cache_path=provider_cache_path,
            holdings_fingerprint_value=current_holdings_fingerprint,
        )
        metrics.record_cache(
            "nasdaq_background_refresh",
            "queued" if background_refresh_queued else "already_running",
        )
    elif selection_refresh_needed:
        metrics.record_cache(
            "nasdaq_background_refresh",
            "disabled",
        )
    database.update_snapshot_observability(
        snapshot_id,
        performance_status=refresh_performance_status,
        timings_ms=metrics.timings_ms,
        cache_statuses=metrics.cache_statuses,
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
        active_share=active_share,
        performance_status=refresh_performance_status,
        timings_ms=dict(metrics.timings_ms),
        cache_statuses=dict(metrics.cache_statuses),
        background_refresh_queued=background_refresh_queued,
    )


def _background_refresh_enabled() -> bool:
    return os.getenv("NDX_BACKGROUND_REFRESH_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _schedule_nasdaq_refresh(
    *,
    db_path: Path,
    holdings: pd.DataFrame,
    holdings_csv: str | Path | None,
    universe: str,
    weighting_basis: str,
    provider_cache_path: str | Path,
    holdings_fingerprint_value: str,
) -> bool:
    job_key = background_job_key(universe, weighting_basis)
    holdings_copy = holdings.copy(deep=True)
    return submit_unique(
        job_key,
        lambda: _run_nasdaq_refresh(
            job_key=job_key,
            db_path=db_path,
            holdings=holdings_copy,
            holdings_csv=holdings_csv,
            universe=universe,
            weighting_basis=weighting_basis,
            provider_cache_path=provider_cache_path,
            holdings_fingerprint_value=holdings_fingerprint_value,
        ),
    )


def _run_nasdaq_refresh(
    *,
    job_key: str,
    db_path: Path,
    holdings: pd.DataFrame,
    holdings_csv: str | Path | None,
    universe: str,
    weighting_basis: str,
    provider_cache_path: str | Path,
    holdings_fingerprint_value: str,
) -> None:
    cache = ProviderCache(provider_cache_path)
    cache.set_job_status(job_key, "running")
    try:
        provider = NasdaqPublicUniverseProvider(
            timeout=int(os.getenv("HTTP_TIMEOUT_SECONDS", "30")),
            cache_path=os.getenv(
                "NASDAQ_UNIVERSE_CACHE_PATH",
                "data/nasdaq_public_universe_cache.csv",
            ),
            liquidity_cache_path=os.getenv(
                "NASDAQ_LIQUIDITY_CACHE_PATH",
                "data/nasdaq_liquidity_cache.csv",
            ),
            cache_ttl_seconds=float(
                os.getenv("NASDAQ_SOURCE_CACHE_TTL_SECONDS", "86400")
            ),
        )
        selection = provider.get_annual_selection(holdings)
        cache.save_selection(
            universe,
            holdings_fingerprint_value,
            selection,
        )
        outcome = recompute_snapshot(
            db_path=db_path,
            holdings_csv=holdings_csv,
            universe=universe,
            weighting_basis=weighting_basis,
            background_refresh=False,
        )
        cache.set_job_status(
            job_key,
            "complete",
            snapshot_id=outcome.snapshot_id,
        )
        LOGGER.info(
            structured_event(
                "nasdaq_background_refresh_complete",
                job_key=job_key,
                snapshot_id=outcome.snapshot_id,
                universe_cache=provider.cache_status,
                liquidity_cache=provider.liquidity_cache_status,
            )
        )
    except Exception as exc:
        cache.set_job_status(
            job_key,
            "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        LOGGER.error(
            structured_event(
                "nasdaq_background_refresh_failed",
                job_key=job_key,
                error_type=type(exc).__name__,
                error=str(exc),
            )
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


def build_spx_holdings_chain(
    universe: str,
    *,
    holdings_csv: str | Path | None = None,
) -> HoldingsProviderChain:
    if universe not in UNIVERSES:
        raise ValueError(f"universe must be one of: {UNIVERSES}.")
    timeout = int(os.getenv("HTTP_TIMEOUT_SECONDS", "30"))
    configured_csv = holdings_csv or os.getenv(
        (
            "NON_UCITS_SPX_HOLDINGS_CSV"
            if universe == "non_ucits"
            else "UCITS_SPX_HOLDINGS_CSV"
        )
    )
    providers = []
    if configured_csv:
        providers.append(
            CsvHoldingsProvider(
                configured_csv,
                source_name=f"configured_{universe}_spx_csv",
                reference_fund="configured_spx_csv",
            )
        )
    if universe == "non_ucits":
        providers.append(
            HttpCsvHoldingsProvider(
                url=os.getenv("IVV_HOLDINGS_URL", DEFAULT_IVV_URL),
                timeout=timeout,
                source_name="ishares_ivv_public_holdings",
                reference_fund="IVV",
            )
        )
    else:
        providers.append(
            HttpCsvHoldingsProvider(
                url=os.getenv("CSPX_HOLDINGS_URL", DEFAULT_CSPX_URL),
                timeout=timeout,
                source_name="ishares_cspx_public_holdings",
                reference_fund="CSPX",
            )
        )
    return HoldingsProviderChain(
        providers,
        min_constituents=450,
        max_constituents=550,
    )


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


def _build_active_share_ndx_holdings(
    holdings: pd.DataFrame,
    rebalance_components: pd.DataFrame | None,
) -> pd.DataFrame:
    """Keep published ETF weights separate from WDI coverage normalization."""
    current = holdings[["ticker", "company_name", "actual_weight"]].copy()
    current["ticker"] = current["ticker"].astype("string").str.upper().str.strip()
    if (
        rebalance_components is None
        or "rebalance_weight" not in rebalance_components
    ):
        return current

    rebalanced = rebalance_components[
        ["ticker", "company_name", "rebalance_weight"]
    ].copy()
    rebalanced["ticker"] = (
        rebalanced["ticker"].astype("string").str.upper().str.strip()
    )
    result = current.merge(
        rebalanced,
        on="ticker",
        how="outer",
        suffixes=("_current", "_rebalanced"),
    )
    result["company_name"] = result["company_name_current"].fillna(
        result["company_name_rebalanced"]
    )
    return result[
        ["ticker", "company_name", "actual_weight", "rebalance_weight"]
    ]
