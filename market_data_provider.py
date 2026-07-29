"""Market-data providers for prices and share counts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import pandas as pd


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
    source_name: str = "yfinance"

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
        normalized = _allocate_shared_float_shares(_normalize_market_data(pd.DataFrame(rows)))
        return _apply_share_overrides(normalized)


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


def evaluate_float_observations(
    frame: pd.DataFrame,
    *,
    float_shares_tolerance: float = 1.10,
    float_cap_tolerance: float = 1.25,
) -> pd.DataFrame:
    """Return common validity masks for every float-based provider path."""
    data = frame.copy()
    for column in ["price", "float_shares", "shares_outstanding", "market_cap"]:
        if column not in data:
            data[column] = pd.NA
        data[column] = pd.to_numeric(data[column], errors="coerce")

    quality = pd.DataFrame(index=data.index)
    quality["price_valid"] = data["price"].notna() & data["price"].gt(0)
    quality["float_valid"] = (
        data["float_shares"].notna() & data["float_shares"].gt(0)
    )
    quality["outstanding_valid"] = (
        data["shares_outstanding"].notna()
        & data["shares_outstanding"].gt(0)
    )
    quality["market_cap_valid"] = (
        data["market_cap"].notna() & data["market_cap"].gt(0)
    )
    quality["float_cap"] = data["price"] * data["float_shares"]
    quality["inconsistent"] = (
        quality["float_valid"]
        & quality["outstanding_valid"]
        & (
            data["float_shares"]
            > data["shares_outstanding"] * float_shares_tolerance
        )
    ) | (
        quality["price_valid"]
        & quality["float_valid"]
        & quality["market_cap_valid"]
        & (
            quality["float_cap"]
            > data["market_cap"] * float_cap_tolerance
        )
    )
    quality["valid"] = (
        quality["price_valid"]
        & quality["float_valid"]
        & ~quality["inconsistent"]
    )
    return quality


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
