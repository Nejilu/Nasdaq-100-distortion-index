"""Persistent stale-while-revalidate wrappers for holdings providers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Protocol, Sequence

import pandas as pd

from acwi_weights_provider import (
    IsharesAcwiFloatWeightsProvider,
    add_yfinance_fallbacks,
    match_acwi_holdings,
)
from provider_cache import CachedProviderFrame, ProviderCache


class FrameProvider(Protocol):
    source_name: str

    def get_holdings(self) -> pd.DataFrame: ...


@dataclass
class CachedHoldingsProvider:
    """Cache a complete parsed provider frame and retain stale data on errors."""

    provider: FrameProvider
    cache: ProviderCache
    cache_key: str
    ttl_seconds: float
    source_name: str = field(init=False)
    reference_fund: str = field(init=False)
    holdings_as_of: str | None = field(default=None, init=False)
    failures: tuple[str, ...] = field(default=(), init=False)
    cache_status: str = field(default="not_checked", init=False)

    def __post_init__(self) -> None:
        self.source_name = str(
            getattr(self.provider, "source_name", "unresolved")
        )
        self.reference_fund = str(
            getattr(self.provider, "reference_fund", "unresolved")
        )

    def get_holdings(self) -> pd.DataFrame:
        cached = self.cache.get_provider_frame(
            self.cache_key,
            max_age_seconds=self.ttl_seconds,
        )
        if cached is not None and cached.is_fresh:
            self.cache_status = "fresh_hit"
            return self._restore(cached)

        try:
            frame = self.provider.get_holdings()
        except Exception as exc:
            if cached is None:
                self.cache_status = "miss_error"
                raise
            self.cache_status = "stale_if_error_hit"
            frame = self._restore(cached)
            self.failures = (
                *self.failures,
                f"live refresh: {type(exc).__name__}: {exc}",
            )
            return frame

        self.source_name = str(
            getattr(self.provider, "source_name", self.source_name)
        )
        self.reference_fund = str(
            getattr(self.provider, "reference_fund", self.reference_fund)
        )
        self.holdings_as_of = getattr(
            self.provider,
            "holdings_as_of",
            frame.attrs.get("holdings_as_of"),
        )
        self.failures = tuple(getattr(self.provider, "failures", ()))
        self.cache.save_provider_frame(
            self.cache_key,
            frame,
            source_name=self.source_name,
            reference_fund=self.reference_fund,
            holdings_as_of=self.holdings_as_of,
            failures=self.failures,
        )
        self.cache_status = (
            "stale_network_refresh" if cached is not None else "miss_network_refresh"
        )
        return frame

    def _restore(self, cached: CachedProviderFrame) -> pd.DataFrame:
        if cached.source_name:
            self.source_name = cached.source_name
        if cached.reference_fund:
            self.reference_fund = cached.reference_fund
        self.holdings_as_of = cached.holdings_as_of
        self.failures = cached.failures
        return cached.frame


@dataclass
class CachedAcwiFloatWeightsProvider:
    """Apply ACWI matching to a persistently cached full ACWI portfolio."""

    provider: IsharesAcwiFloatWeightsProvider
    cache: ProviderCache
    cache_key: str
    ttl_seconds: float
    _holdings: CachedHoldingsProvider = field(init=False)

    def __post_init__(self) -> None:
        self._holdings = CachedHoldingsProvider(
            provider=self.provider,
            cache=self.cache,
            cache_key=self.cache_key,
            ttl_seconds=self.ttl_seconds,
        )

    @property
    def source_name(self) -> str:
        return self._holdings.source_name

    @property
    def holdings_as_of(self) -> str | None:
        return self._holdings.holdings_as_of

    @property
    def cache_status(self) -> str:
        return self._holdings.cache_status

    def build_reference(
        self,
        holdings: pd.DataFrame,
        market_data: pd.DataFrame,
        *,
        adr_tickers: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        acwi = self._holdings.get_holdings()
        references = match_acwi_holdings(
            holdings,
            acwi,
            adr_tickers=adr_tickers,
        )
        return add_yfinance_fallbacks(
            references,
            market_data,
            min_calibration_rows=self.provider.min_calibration_rows,
        )


def provider_cache_key(namespace: str, provider: object) -> str:
    """Build a stable key that changes with provider URLs and local inputs."""
    descriptor = _provider_descriptor(provider)
    digest = hashlib.sha256(
        json.dumps(
            descriptor,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:20]
    return f"{namespace}:{digest}"


def _provider_descriptor(provider: object) -> object:
    if isinstance(provider, (str, int, float, bool)) or provider is None:
        return provider
    if isinstance(provider, Path):
        resolved = provider.expanduser().resolve()
        modified = resolved.stat().st_mtime_ns if resolved.exists() else None
        return {"path": str(resolved), "modified": modified}
    if isinstance(provider, (list, tuple)):
        return [_provider_descriptor(value) for value in provider]
    if is_dataclass(provider):
        ignored = {
            "cache_status",
            "failures",
            "holdings_as_of",
            "liquidity_cache_status",
            "source_name",
        }
        return {
            "class": type(provider).__name__,
            "fields": {
                item.name: _provider_descriptor(getattr(provider, item.name))
                for item in fields(provider)
                if item.name not in ignored and item.init
            },
        }
    return {"class": type(provider).__name__, "value": str(provider)}
