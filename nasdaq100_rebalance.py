"""Nasdaq-100 reconstitution and modified-cap weighting simulation.

The calculation rules follow the Nasdaq-100 methodology effective May 1, 2026.
Public Nasdaq, SEC and Yahoo data are used for the live eligibility screen.
Nasdaq retains discretion and does not publish every index-review input, so the
composition result is an auditable public-data simulation rather than an
official index announcement.
"""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
from dataclasses import dataclass, field
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from ndx_wdi.domain.rebalance import (  # noqa: F401
    RebalanceResult,
    SelectionResult,
    _normalize_positive,
    _proportional_with_caps,
    apply_annual_security_capping,
    apply_company_capping,
    build_quarterly_index_share_weights,
    company_capping_required,
    derive_acwi_total_cap_conversion,
    fallback_current_selection,
    select_annual_companies,
    select_annual_universe,
    select_quarterly_companies,
    simulate_rebalance,
)


METHODOLOGY_VERSION = "Nasdaq-100 methodology effective 2026-05-01"
METHODOLOGY_URL = "https://indexes.nasdaqomx.com/docs/Methodology_NDX.pdf"
WEIGHT_CALCULATION_URL = (
    "https://indexes.nasdaqomx.com/docs/Nasdaq_Index_Weight_Calculations.pdf"
)
NASDAQ_SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks"
NASDAQ_SYMBOL_DIRECTORY_URL = (
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
)
SEC_TICKER_EXCHANGE_URL = (
    "https://www.sec.gov/files/company_tickers_exchange.json"
)

NON_EQUITY_NAME_PATTERN = re.compile(
    r"\b(?:WARRANTS?|RIGHTS?|UNITS?|PREFERRED|NOTES?\s+DUE|BONDS?|"
    r"BENEFICIAL\s+INTEREST|DEPOSITARY\s+SHARES?.*PREFERRED)\b",
    re.IGNORECASE,
)
REIT_OR_SPAC_PATTERN = re.compile(
    r"\b(?:REIT|ACQUISITION\s+(?:CORP|CORPORATION|COMPANY)|BLANK\s+CHECK)\b",
    re.IGNORECASE,
)


@dataclass
class NasdaqPublicUniverseProvider:
    """Build a daily eligibility universe from public Nasdaq and SEC files."""

    timeout: int = 45
    cache_path: str | Path = "data/nasdaq_public_universe_cache.csv"
    liquidity_cache_path: str | Path = "data/nasdaq_liquidity_cache.csv"
    cache_ttl_seconds: float = 24 * 60 * 60
    source_name: str = "nasdaq_public_screener+symbol_directory+sec_cik+yfinance"
    cache_status: str = field(default="not_checked", init=False)
    liquidity_cache_status: str = field(default="not_checked", init=False)

    def get_quarterly_selection(
        self,
        current_holdings: pd.DataFrame,
        *,
        as_of: date | None = None,
    ) -> SelectionResult:
        as_of = as_of or date.today()
        current_tickers = {
            str(value).upper().strip()
            for value in current_holdings["ticker"].dropna()
        }
        universe = self._download_universe()
        universe["is_current"] = universe["ticker"].isin(current_tickers)

        missing_current = current_tickers.difference(universe["ticker"])
        if missing_current:
            fallback_names = (
                current_holdings.set_index("ticker")["company_name"].to_dict()
                if "company_name" in current_holdings
                else {}
            )
            universe = pd.concat(
                [
                    universe,
                    pd.DataFrame(
                        {
                            "ticker": sorted(missing_current),
                            "company_name": [
                                fallback_names.get(ticker, ticker)
                                for ticker in sorted(missing_current)
                            ],
                            "company_id": [
                                f"TICKER:{ticker}" for ticker in sorted(missing_current)
                            ],
                            "full_market_cap": np.nan,
                            "security_type": "Ordinary share",
                            "is_current": True,
                            "base_eligible": True,
                            "ipo_year": np.nan,
                        }
                    ),
                ],
                ignore_index=True,
            )

        company_cap = universe.groupby("company_id")["full_market_cap"].transform("max")
        universe["company_full_market_cap"] = company_cap
        candidates = (
            universe.loc[universe["base_eligible"] | universe["is_current"]]
            .sort_values("company_full_market_cap", ascending=False, na_position="last")
            .drop_duplicates("company_id")
            .head(180)
        )
        liquidity = self._download_liquidity(
            candidates["ticker"].tolist(),
            as_of=as_of,
        )
        universe = universe.merge(liquidity, on="ticker", how="left")
        _require_known_liquidity(universe, candidates["ticker"])

        current = universe["is_current"]
        liquid = universe["advt_3m"].ge(5_000_000)
        seasoned = universe["first_trade_date"].map(
            lambda value: _has_three_full_calendar_months(value, as_of)
        )
        # The public screener reports global company capitalization for foreign
        # ADRs, while the methodology uses listed depositary shares for a
        # non-primary ADR. Do not admit a new ADR without that unavailable
        # primary/listed-share verification.
        verified_listing_form = current | ~universe["security_type"].eq("ADR/ADS")
        universe["regular_eligible"] = (
            universe["base_eligible"]
            & verified_listing_form
            & liquid
            & (seasoned | current)
        )
        universe["fast_entry_eligible"] = (
            universe["base_eligible"]
            & verified_listing_form
            & liquid
            & universe["trading_days"].ge(15)
        )

        result = select_quarterly_companies(universe, current_tickers)
        notes = list(result.notes)
        notes.append(
            "Composition uses public Nasdaq/SEC classifications and yfinance "
            "liquidity; Nasdaq discretion and non-public review inputs are not "
            "replicable."
        )
        notes.append(
            "Non-constituent ADRs are excluded unless primary-listing and listed "
            "depositary-share inputs can be verified."
        )
        return SelectionResult(
            securities=result.securities,
            selected_tickers=result.selected_tickers,
            selected_company_ids=result.selected_company_ids,
            additions=result.additions,
            removals=result.removals,
            status="public_data_simulation",
            source=self.source_name,
            as_of=as_of.isoformat(),
            eligible_company_count=result.eligible_company_count,
            notes=tuple(notes),
        )

    def get_annual_selection(
        self,
        current_holdings: pd.DataFrame,
        *,
        as_of: date | None = None,
    ) -> SelectionResult:
        """Simulate the December reconstitution with current public inputs."""
        as_of = as_of or date.today()
        current_tickers = {
            str(value).upper().strip()
            for value in current_holdings["ticker"].dropna()
        }
        universe = self._download_universe()
        universe["is_current"] = universe["ticker"].isin(current_tickers)

        missing_current = current_tickers.difference(universe["ticker"])
        if missing_current:
            fallback_names = (
                current_holdings.set_index("ticker")["company_name"].to_dict()
                if "company_name" in current_holdings
                else {}
            )
            universe = pd.concat(
                [
                    universe,
                    pd.DataFrame(
                        {
                            "ticker": sorted(missing_current),
                            "company_name": [
                                fallback_names.get(ticker, ticker)
                                for ticker in sorted(missing_current)
                            ],
                            "company_id": [
                                f"TICKER:{ticker}" for ticker in sorted(missing_current)
                            ],
                            "full_market_cap": np.nan,
                            "security_type": "Ordinary share",
                            "is_current": True,
                            "base_eligible": True,
                            "ipo_year": np.nan,
                        }
                    ),
                ],
                ignore_index=True,
            )

        universe["company_full_market_cap"] = universe.groupby("company_id")[
            "full_market_cap"
        ].transform("max")
        candidate_company_ids = (
            universe.loc[universe["base_eligible"] | universe["is_current"]]
            .sort_values("company_full_market_cap", ascending=False, na_position="last")
            .drop_duplicates("company_id")
            .head(180)
            ["company_id"]
        )
        candidate_securities = universe.loc[
            universe["company_id"].isin(candidate_company_ids)
            & (universe["base_eligible"] | universe["is_current"]),
            "ticker",
        ]
        liquidity = self._download_liquidity(
            candidate_securities.tolist(),
            as_of=as_of,
        )
        universe = universe.merge(liquidity, on="ticker", how="left")
        _require_known_liquidity(universe, candidate_securities)

        current = universe["is_current"]
        liquid = universe["advt_3m"].ge(5_000_000)
        seasoned = universe["first_trade_date"].map(
            lambda value: _has_three_full_calendar_months(value, as_of)
        )
        verified_listing_form = current | ~universe["security_type"].eq("ADR/ADS")
        universe["regular_eligible"] = (
            universe["base_eligible"]
            & verified_listing_form
            & liquid
            & (seasoned | current)
        )

        result = select_annual_universe(universe, current_tickers)
        notes = list(result.notes)
        notes.extend(
            [
                "Composition uses public Nasdaq/SEC classifications and "
                "yfinance liquidity; Nasdaq discretion and non-public review "
                "inputs are not replicable.",
                "Non-constituent ADRs are excluded unless primary-listing and "
                "listed depositary-share inputs can be verified.",
                "Current constituent status is used as a conservative public "
                "proxy for the prior reconstitution top-100 and subsequent "
                "addition flags.",
            ]
        )
        return SelectionResult(
            securities=result.securities,
            selected_tickers=result.selected_tickers,
            selected_company_ids=result.selected_company_ids,
            additions=result.additions,
            removals=result.removals,
            status="public_data_simulation",
            source=self.source_name,
            as_of=as_of.isoformat(),
            eligible_company_count=result.eligible_company_count,
            notes=tuple(notes),
        )

    def _download_universe(self) -> pd.DataFrame:
        fresh_cache = self._load_universe_cache(require_fresh=True)
        if fresh_cache is not None:
            self.cache_status = "fresh_hit"
            return fresh_cache
        browser_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/127.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": "https://www.nasdaq.com",
            "Referer": "https://www.nasdaq.com/market-activity/stocks/screener",
        }
        rows: list[dict[str, object]] = []
        last_error: Exception | None = None
        for attempt in range(3):
            params: dict[str, object] = {
                "tableonly": "true",
                "limit": 5000 if attempt < 2 else 10000,
                "exchange": "nasdaq",
            }
            if attempt != 1:
                params.update({"download": "true", "offset": 0})
            try:
                screener_response = requests.get(
                    NASDAQ_SCREENER_URL,
                    params=params,
                    headers=browser_headers,
                    timeout=self.timeout,
                )
                screener_response.raise_for_status()
                payload = screener_response.json()
                response_data = payload.get("data") or {}
                rows = (
                    response_data.get("rows")
                    or (response_data.get("table") or {}).get("rows")
                    or []
                )
                if len(rows) >= 1_000:
                    break
            except (requests.RequestException, ValueError, TypeError) as exc:
                last_error = exc
            if attempt < 2:
                time.sleep(1.0 + attempt)
        if len(rows) < 1_000:
            cached = self._load_universe_cache(require_fresh=False)
            if cached is not None:
                self.cache_status = "stale_if_error_hit"
                return cached
            self.cache_status = "miss"
            detail = f" Last error: {last_error}" if last_error else ""
            raise ValueError(
                f"Nasdaq screener returned only {len(rows)} rows.{detail}"
            )
        screener = pd.DataFrame(rows).rename(
            columns={
                "symbol": "ticker",
                "name": "company_name",
                "marketCap": "full_market_cap",
                "ipoyear": "ipo_year",
            }
        )
        screener["ticker"] = screener["ticker"].astype("string").str.upper().str.strip()
        screener["full_market_cap"] = pd.to_numeric(
            screener["full_market_cap"], errors="coerce"
        )
        screener["ipo_year"] = pd.to_numeric(screener["ipo_year"], errors="coerce")

        symbols_response = requests.get(
            NASDAQ_SYMBOL_DIRECTORY_URL,
            timeout=self.timeout,
            headers={"User-Agent": "NDX-WDI/1.0"},
        )
        symbols_response.raise_for_status()
        symbols = pd.read_csv(StringIO(symbols_response.text), sep="|")
        symbols = symbols.loc[
            symbols["Symbol"].astype("string").ne("File Creation Time")
        ].rename(columns={"Symbol": "ticker"})
        symbols["ticker"] = symbols["ticker"].astype("string").str.upper().str.strip()

        sec_response = requests.get(
            SEC_TICKER_EXCHANGE_URL,
            timeout=self.timeout,
            headers={
                "User-Agent": os.getenv(
                    "SEC_USER_AGENT",
                    "NDX-WDI research ndx-wdi@example.com",
                )
            },
        )
        sec_response.raise_for_status()
        sec_payload = sec_response.json()
        sec = pd.DataFrame(sec_payload["data"], columns=sec_payload["fields"])
        sec["ticker"] = sec["ticker"].astype("string").str.upper().str.strip()
        sec = sec.drop_duplicates("ticker", keep="first")

        universe = (
            screener.merge(symbols, on="ticker", how="inner")
            .merge(sec[["ticker", "cik"]], on="ticker", how="left")
        )
        universe["company_id"] = universe["cik"].map(
            lambda value: (
                f"CIK:{int(value)}" if pd.notna(value) else None
            )
        )
        missing_company = universe["company_id"].isna()
        universe.loc[missing_company, "company_id"] = universe.loc[
            missing_company, "ticker"
        ].map(lambda value: f"TICKER:{value}")

        name = universe["company_name"].fillna("")
        industry = universe.get("industry", pd.Series("", index=universe.index)).fillna("")
        sector = universe.get("sector", pd.Series("", index=universe.index)).fillna("")
        ordinary_or_adr = ~name.str.contains(NON_EQUITY_NAME_PATTERN, na=False)
        excluded_structure = (
            name.str.contains(REIT_OR_SPAC_PATTERN, na=False)
            | industry.str.contains(REIT_OR_SPAC_PATTERN, na=False)
        )
        universe["base_eligible"] = (
            universe["Market Category"].isin(["Q", "G"])
            & universe["ETF"].eq("N")
            & universe["Test Issue"].eq("N")
            & universe["Financial Status"].eq("N")
            & universe["full_market_cap"].gt(0)
            & ~sector.str.casefold().eq("finance")
            & ordinary_or_adr
            & ~excluded_structure
        )
        universe["security_type"] = np.where(
            name.str.contains(
                r"\b(?:ADR|ADS|AMERICAN DEPOSITARY|NEW YORK REGISTRY)\b",
                case=False,
                regex=True,
                na=False,
            ),
            "ADR/ADS",
            "Ordinary share",
        )
        result = universe[
            [
                "ticker",
                "company_name",
                "company_id",
                "full_market_cap",
                "sector",
                "industry",
                "ipo_year",
                "security_type",
                "base_eligible",
            ]
        ].drop_duplicates("ticker", keep="first")
        cache_path = Path(self.cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache_path, index=False)
        self.cache_status = "network_refresh"
        return result

    def _load_universe_cache(
        self,
        *,
        require_fresh: bool,
    ) -> pd.DataFrame | None:
        cache_path = Path(self.cache_path)
        if not cache_path.exists():
            return None
        if require_fresh and not _path_is_fresh(
            cache_path,
            self.cache_ttl_seconds,
        ):
            return None
        cached = pd.read_csv(cache_path)
        required = {
            "ticker",
            "company_name",
            "company_id",
            "full_market_cap",
            "security_type",
            "base_eligible",
        }
        if not required.issubset(cached.columns) or len(cached) < 1_000:
            return None
        cached["base_eligible"] = cached["base_eligible"].astype(bool)
        return cached

    def _download_liquidity(
        self,
        tickers: list[str],
        *,
        as_of: date,
    ) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "advt_3m",
                    "first_trade_date",
                    "trading_days",
                    "liquidity_status",
                ]
            )
        import yfinance as yf

        tickers = list(dict.fromkeys(tickers))
        cached = self._load_liquidity_cache(tickers, require_fresh=True)
        if cached is not None:
            self.liquidity_cache_status = "fresh_hit"
            return cached
        start = pd.Timestamp(as_of) - pd.DateOffset(months=5)
        end = pd.Timestamp(as_of) + pd.Timedelta(days=1)
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            yf.download,
            tickers=tickers,
            start=start.date().isoformat(),
            end=end.date().isoformat(),
            auto_adjust=False,
            progress=False,
            threads=min(8, len(tickers)),
            group_by="column",
            timeout=max(1, min(int(self.timeout), 30)),
        )
        try:
            history = future.result(timeout=max(float(self.timeout), 0.01))
        except FuturesTimeoutError as exc:
            future.cancel()
            cached = self._load_liquidity_cache(
                tickers,
                require_fresh=False,
            )
            if cached is not None:
                self.liquidity_cache_status = "stale_if_error_hit"
                return cached
            self.liquidity_cache_status = "miss"
            raise TimeoutError(
                f"yfinance liquidity deadline exceeded after {self.timeout}s"
            ) from exc
        except Exception:
            cached = self._load_liquidity_cache(
                tickers,
                require_fresh=False,
            )
            if cached is not None:
                self.liquidity_cache_status = "stale_if_error_hit"
                return cached
            self.liquidity_cache_status = "miss"
            raise
        finally:
            executor.shutdown(wait=future.done(), cancel_futures=True)
        rows: list[dict[str, object]] = []
        cutoff = pd.Timestamp(as_of) - pd.DateOffset(months=3)
        for ticker in tickers:
            try:
                close = (
                    history["Close"][ticker]
                    if isinstance(history.columns, pd.MultiIndex)
                    else history["Close"]
                )
                volume = (
                    history["Volume"][ticker]
                    if isinstance(history.columns, pd.MultiIndex)
                    else history["Volume"]
                )
            except (KeyError, TypeError):
                close = pd.Series(dtype="float64")
                volume = pd.Series(dtype="float64")
            valid = pd.DataFrame({"close": close, "volume": volume}).dropna()
            recent = valid.loc[valid.index.tz_localize(None) >= cutoff]
            rows.append(
                {
                    "ticker": ticker,
                    "advt_3m": (
                        float((recent["close"] * recent["volume"]).mean())
                        if not recent.empty
                        else np.nan
                    ),
                    "first_trade_date": (
                        valid.index.min().tz_localize(None).date().isoformat()
                        if not valid.empty
                        else None
                    ),
                    "trading_days": int(len(valid)),
                    "liquidity_status": (
                        "available" if not recent.empty else "unknown"
                    ),
                }
            )
        result = pd.DataFrame(rows)
        cache_path = Path(self.liquidity_cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache_path, index=False)
        self.liquidity_cache_status = "network_refresh"
        return result

    def _load_liquidity_cache(
        self,
        tickers: list[str],
        *,
        require_fresh: bool,
    ) -> pd.DataFrame | None:
        cache_path = Path(self.liquidity_cache_path)
        if not cache_path.exists():
            return None
        if require_fresh and not _path_is_fresh(
            cache_path,
            self.cache_ttl_seconds,
        ):
            return None
        cached = pd.read_csv(cache_path)
        required = {
            "ticker",
            "advt_3m",
            "first_trade_date",
            "trading_days",
            "liquidity_status",
        }
        if not required.issubset(cached.columns):
            return None
        cached["ticker"] = cached["ticker"].astype("string").str.upper()
        requested = set(tickers)
        available = set(cached["ticker"].dropna().astype(str))
        if not requested.issubset(available):
            return None
        return cached.loc[cached["ticker"].isin(requested)].reset_index(
            drop=True
        )


def _require_known_liquidity(
    universe: pd.DataFrame,
    candidate_tickers: Iterable[str],
) -> None:
    """Abort selection instead of treating missing provider data as a failed rule."""
    requested = {
        str(ticker).upper().strip()
        for ticker in candidate_tickers
        if str(ticker).strip()
    }
    status = universe.get(
        "liquidity_status",
        pd.Series("unknown", index=universe.index),
    ).astype("string")
    advt = pd.to_numeric(
        universe.get("advt_3m", pd.Series(np.nan, index=universe.index)),
        errors="coerce",
    )
    observed = set(
        universe.loc[
            universe["ticker"].isin(requested)
            & status.eq("available")
            & advt.notna(),
            "ticker",
        ].astype(str)
    )
    unknown = sorted(requested.difference(observed))
    if unknown:
        preview = ", ".join(unknown[:10])
        suffix = "" if len(unknown) <= 10 else f", and {len(unknown) - 10} more"
        raise ValueError(
            "Liquidity eligibility is unknown for "
            f"{len(unknown)} candidate securities: {preview}{suffix}."
        )


def _path_is_fresh(path: Path, max_age_seconds: float) -> bool:
    if max_age_seconds <= 0:
        return False
    try:
        age_seconds = max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return False
    return age_seconds <= max_age_seconds

def _has_three_full_calendar_months(
    first_trade_date: object,
    as_of: date,
) -> bool:
    parsed = pd.to_datetime(first_trade_date, errors="coerce")
    if pd.isna(parsed):
        return False
    first_eligible_month = (
        parsed.to_period("M").to_timestamp() + pd.DateOffset(months=4)
    )
    return pd.Timestamp(as_of) >= first_eligible_month
