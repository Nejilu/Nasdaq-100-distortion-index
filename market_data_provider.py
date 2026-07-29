"""Market-data providers for prices and share counts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Protocol, Sequence

import numpy as np
import pandas as pd

from ndx_wdi.domain.market_quality import evaluate_float_observations  # noqa: F401
from provider_cache import ProviderCache


FLOAT_SHARES_OVERRIDES = {
    # Nasdaq-listed ASML ADR float. Do not substitute the Dutch ordinary-share
    # float or yfinance's inconsistent consolidated value.
    "ASML": 88_000_000.0,
}
FLOAT_SHARES_OVERRIDE_STATUS = "hardcoded_float_override"
TOTAL_SHARES_OVERRIDES = {
    # The Nasdaq listing uses the same maintained ASML ADR share count. Keeping
    # total and floating shares aligned prevents an unsupported automatic 3x.
    "ASML": 88_000_000.0,
}
TOTAL_SHARES_OVERRIDE_STATUS = "hardcoded_total_shares_override"


class MarketDataProvider(Protocol):
    """Contract implemented by market data sources."""

    source_name: str

    def get_market_data(self, tickers: Sequence[str]) -> pd.DataFrame:
        """Return price/share data plus fields used for consistency checks."""


@dataclass
class YFinanceMarketDataProvider:
    """Retrieve current price and share counts from yfinance."""

    max_workers: int = 8
    batch_timeout_seconds: float = 60.0
    cache_dir: str | Path = "data/yfinance_cache"
    persistent_cache_path: str | Path | None = None
    price_ttl_seconds: float = 10 * 60
    fundamentals_ttl_seconds: float = 24 * 60 * 60
    failure_retry_seconds: float = 60
    source_name: str = "yfinance"
    cache_status: str = "disabled"

    def _configure_cache(self) -> None:
        """Keep yfinance SQLite caches inside the writable project directory."""
        import yfinance as yf

        cache_path = Path(self.cache_dir).resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache_path))

    @staticmethod
    def _fetch_one(ticker: str) -> dict[str, object]:
        # Import lazily so pure calculation tests do not initialize yfinance.
        import yfinance as yf

        instrument = yf.Ticker(ticker)
        info = instrument.get_info()
        price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        return {
            "ticker": ticker,
            "price": price,
            "float_shares": info.get("floatShares"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "market_cap": info.get("marketCap"),
            "float_shares_status": "reported",
            "market_data_error": None,
        }

    def get_market_data(self, tickers: Sequence[str]) -> pd.DataFrame:
        self._configure_cache()
        unique_tickers = list(dict.fromkeys(str(ticker).upper() for ticker in tickers))
        if self.persistent_cache_path is not None:
            rows = self._get_cached_market_data(unique_tickers)
        else:
            rows = self._fetch_many(unique_tickers)
        normalized = _allocate_shared_float_shares(
            _normalize_market_data(pd.DataFrame(rows))
        )
        return _apply_share_overrides(normalized)

    def _fetch_many(self, tickers: Sequence[str]) -> list[dict[str, object]]:
        unique_tickers = list(dict.fromkeys(str(ticker).upper() for ticker in tickers))
        rows: list[dict[str, object]] = []
        worker_count = max(1, min(int(self.max_workers), max(len(unique_tickers), 1)))
        executor = ThreadPoolExecutor(max_workers=worker_count)
        try:
            futures = {executor.submit(self._fetch_one, ticker): ticker for ticker in unique_tickers}
            completed, pending = wait(
                futures,
                timeout=max(float(self.batch_timeout_seconds), 0.01),
            )
            for future in completed:
                ticker = futures[future]
                try:
                    rows.append(future.result())
                except Exception as exc:  # one bad ticker must not invalidate the snapshot
                    rows.append(_missing_market_data_row(ticker, str(exc)))
            for future in pending:
                ticker = futures[future]
                future.cancel()
                rows.append(
                    _missing_market_data_row(
                        ticker,
                        "yfinance batch deadline exceeded",
                    )
                )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        return rows

    @staticmethod
    def _fetch_prices(tickers: Sequence[str]) -> dict[str, float]:
        if not tickers:
            return {}
        import yfinance as yf

        unique_tickers = list(dict.fromkeys(tickers))
        history = yf.download(
            tickers=unique_tickers,
            period="5d",
            auto_adjust=False,
            progress=False,
            threads=min(8, len(unique_tickers)),
            group_by="column",
            timeout=10,
        )
        if history.empty or "Close" not in history:
            return {}
        close = history["Close"]
        prices: dict[str, float] = {}
        if isinstance(close, pd.Series):
            valid = pd.to_numeric(close, errors="coerce").dropna()
            if not valid.empty and len(unique_tickers) == 1:
                prices[unique_tickers[0]] = float(valid.iloc[-1])
            return prices
        for ticker in unique_tickers:
            if ticker not in close:
                continue
            valid = pd.to_numeric(close[ticker], errors="coerce").dropna()
            if not valid.empty:
                prices[ticker] = float(valid.iloc[-1])
        return prices

    def _get_cached_market_data(
        self,
        tickers: Sequence[str],
    ) -> list[dict[str, object]]:
        cache = ProviderCache(self.persistent_cache_path)
        cached = cache.get_market_data(tickers)
        cached_by_ticker = (
            cached.set_index("ticker").to_dict(orient="index")
            if not cached.empty
            else {}
        )
        now = time.time()
        fundamentals_to_fetch: list[str] = []
        prices_to_fetch: list[str] = []
        for ticker in tickers:
            row = cached_by_ticker.get(ticker, {})
            fundamentals_age = _age_seconds(
                now,
                row.get("fundamentals_fetched_at"),
            )
            attempt_age = _age_seconds(
                now,
                row.get("fundamentals_attempted_at"),
            )
            price_age = _age_seconds(now, row.get("price_fetched_at"))
            fundamentals_fresh = fundamentals_age <= self.fundamentals_ttl_seconds
            retry_blocked = (
                fundamentals_age > self.fundamentals_ttl_seconds
                and attempt_age <= self.failure_retry_seconds
            )
            if not fundamentals_fresh and not retry_blocked:
                fundamentals_to_fetch.append(ticker)
            if price_age > self.price_ttl_seconds:
                prices_to_fetch.append(ticker)

        price_executor = ThreadPoolExecutor(max_workers=1)
        price_future = (
            price_executor.submit(self._fetch_prices, prices_to_fetch)
            if prices_to_fetch
            else None
        )
        fetched_fundamentals = (
            self._fetch_many(fundamentals_to_fetch)
            if fundamentals_to_fetch
            else []
        )
        fetched_by_ticker = {
            str(row.get("ticker", "")).upper(): row
            for row in fetched_fundamentals
        }
        prices: dict[str, float] = {}
        if price_future is not None:
            try:
                prices = price_future.result(
                    timeout=max(min(self.batch_timeout_seconds, 15.0), 0.01)
                )
            except Exception:
                price_future.cancel()
        price_executor.shutdown(wait=False, cancel_futures=True)

        output_rows: list[dict[str, object]] = []
        for ticker in tickers:
            row = dict(cached_by_ticker.get(ticker, {}))
            row["ticker"] = ticker
            fetched = fetched_by_ticker.get(ticker)
            if fetched is not None:
                row["fundamentals_attempted_at"] = now
                if _has_valid_fundamentals(fetched):
                    for column in [
                        "float_shares",
                        "shares_outstanding",
                        "market_cap",
                        "float_shares_status",
                        "shares_outstanding_status",
                    ]:
                        row[column] = fetched.get(column)
                    row["fundamentals_fetched_at"] = now
                row["market_data_error"] = fetched.get("market_data_error")
                fetched_price = _positive_number(fetched.get("price"))
                if fetched_price is not None:
                    row["price"] = fetched_price
                    row["price_fetched_at"] = now
            if ticker in prices:
                row["price"] = prices[ticker]
                row["price_fetched_at"] = now
            output_rows.append(row)

        cache.upsert_market_data(pd.DataFrame(output_rows))
        cached_count = len(cached_by_ticker)
        network_count = len(
            set(fundamentals_to_fetch).union(prices_to_fetch)
        )
        if network_count == 0:
            self.cache_status = f"fresh_hit:{cached_count}/{len(tickers)}"
        elif cached_count:
            self.cache_status = (
                f"partial_hit:{cached_count}/{len(tickers)};"
                f"network:{network_count}"
            )
        else:
            self.cache_status = f"miss;network:{network_count}"
        return output_rows


def _normalize_market_data(frame: pd.DataFrame) -> pd.DataFrame:
    expected = [
        "ticker",
        "price",
        "float_shares",
        "shares_outstanding",
        "market_cap",
        "float_shares_status",
        "shares_outstanding_status",
        "market_data_error",
    ]
    for column in expected:
        if column not in frame:
            frame[column] = None
    frame["ticker"] = frame["ticker"].astype("string").str.upper().str.strip()
    for column in ["price", "float_shares", "shares_outstanding", "market_cap"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return frame[expected].drop_duplicates("ticker", keep="last").reset_index(drop=True)


def _allocate_shared_float_shares(frame: pd.DataFrame) -> pd.DataFrame:
    """Split one consolidated float across clearly matching share classes.

    Yahoo currently reports the same consolidated Alphabet float for GOOG and
    GOOGL. Allocation is only applied when the duplicate is exact, prices and
    market caps match closely, and the consolidated float is plausible against
    the sum of class-level shares outstanding.
    """
    result = frame.copy()
    candidates = result.loc[
        result["float_shares"].notna() & (result["float_shares"] > 0)
    ].groupby("float_shares", sort=False)
    for reported_float, group in candidates:
        if len(group) < 2:
            continue
        outstanding = group["shares_outstanding"]
        if outstanding.isna().any() or (outstanding <= 0).any():
            continue
        total_outstanding = float(outstanding.sum())
        float_ratio = float(reported_float) / total_outstanding
        if not 0.50 <= float_ratio <= 1.10:
            continue
        if not (float(reported_float) > outstanding * 1.10).all():
            continue
        prices = group["price"].dropna()
        if len(prices) != len(group) or float(prices.max() / prices.min()) > 1.25:
            continue
        market_caps = group["market_cap"].dropna()
        if len(market_caps) == len(group) and float(market_caps.max() / market_caps.min()) > 1.25:
            continue
        allocated = float(reported_float) * outstanding / total_outstanding
        result.loc[group.index, "float_shares"] = allocated
        result.loc[group.index, "float_shares_status"] = "allocated_shared_float"
    return result


def _apply_float_share_overrides(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply maintained listing-specific overrides after provider cleanup."""
    result = frame.copy()
    for ticker, float_shares in FLOAT_SHARES_OVERRIDES.items():
        matches = result["ticker"].eq(ticker)
        result.loc[matches, "float_shares"] = float_shares
        result.loc[matches, "float_shares_status"] = (
            FLOAT_SHARES_OVERRIDE_STATUS
        )
    return result


def _apply_share_overrides(frame: pd.DataFrame) -> pd.DataFrame:
    result = _apply_float_share_overrides(frame)
    for ticker, shares_outstanding in TOTAL_SHARES_OVERRIDES.items():
        matches = result["ticker"].eq(ticker)
        result.loc[matches, "shares_outstanding"] = shares_outstanding
        result.loc[matches, "shares_outstanding_status"] = (
            TOTAL_SHARES_OVERRIDE_STATUS
        )
    return result


def _age_seconds(now: float, fetched_at: object) -> float:
    try:
        value = float(fetched_at)
    except (TypeError, ValueError):
        return float("inf")
    if not np.isfinite(value):
        return float("inf")
    return max(0.0, now - value)


def _positive_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number <= 0:
        return None
    return number


def _has_valid_fundamentals(row: dict[str, object]) -> bool:
    return any(
        _positive_number(row.get(column)) is not None
        for column in ["float_shares", "shares_outstanding", "market_cap"]
    )


def _missing_market_data_row(ticker: str, error: str) -> dict[str, object]:
    return {
        "ticker": ticker,
        "price": None,
        "float_shares": None,
        "shares_outstanding": None,
        "market_cap": None,
        "float_shares_status": None,
        "shares_outstanding_status": None,
        "market_data_error": error,
    }
