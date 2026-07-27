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
from dataclasses import dataclass
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests


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


@dataclass(frozen=True)
class SelectionResult:
    securities: pd.DataFrame
    selected_tickers: tuple[str, ...]
    selected_company_ids: tuple[str, ...]
    additions: tuple[str, ...]
    removals: tuple[str, ...]
    status: str
    source: str
    as_of: str
    eligible_company_count: int
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RebalanceResult:
    components: pd.DataFrame
    ndx_wdi: float
    coverage_ratio: float
    constituent_count: int
    status: str
    method: str
    reference_date: str
    additions: tuple[str, ...]
    removals: tuple[str, ...]
    data_source: str
    acwi_conversion_scale: float | None = None
    acwi_calibration_count: int = 0
    notes: tuple[str, ...] = ()


@dataclass
class NasdaqPublicUniverseProvider:
    """Build a daily eligibility universe from public Nasdaq and SEC files."""

    timeout: int = 45
    cache_path: str | Path = "data/nasdaq_public_universe_cache.csv"
    source_name: str = "nasdaq_public_screener+symbol_directory+sec_cik+yfinance"

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

    def _download_universe(self) -> pd.DataFrame:
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
            cache_path = Path(self.cache_path)
            if cache_path.exists():
                cached = pd.read_csv(cache_path)
                required = {
                    "ticker",
                    "company_name",
                    "company_id",
                    "full_market_cap",
                    "security_type",
                    "base_eligible",
                }
                if required.issubset(cached.columns) and len(cached) >= 1_000:
                    cached["base_eligible"] = cached["base_eligible"].astype(bool)
                    return cached
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
        return result

    @staticmethod
    def _download_liquidity(tickers: list[str], *, as_of: date) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame(
                columns=["ticker", "advt_3m", "first_trade_date", "trading_days"]
            )
        import yfinance as yf

        start = pd.Timestamp(as_of) - pd.DateOffset(months=5)
        end = pd.Timestamp(as_of) + pd.Timedelta(days=1)
        history = yf.download(
            tickers=tickers,
            start=start.date().isoformat(),
            end=end.date().isoformat(),
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="column",
        )
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
                }
            )
        return pd.DataFrame(rows)


def fallback_current_selection(
    holdings: pd.DataFrame,
    *,
    as_of: date | None = None,
    reason: str,
) -> SelectionResult:
    """Preserve current membership when the public universe cannot be loaded."""
    as_of = as_of or date.today()
    securities = holdings[["ticker", "company_name"]].copy()
    securities["ticker"] = securities["ticker"].astype("string").str.upper().str.strip()
    securities["company_id"] = securities["ticker"].map(lambda value: f"TICKER:{value}")
    securities["security_type"] = "Ordinary share"
    securities["is_current"] = True
    securities["selected"] = True
    tickers = tuple(securities["ticker"])
    company_ids = tuple(securities["company_id"])
    return SelectionResult(
        securities=securities,
        selected_tickers=tickers,
        selected_company_ids=company_ids,
        additions=(),
        removals=(),
        status="current_composition_fallback",
        source="current_holdings_only",
        as_of=as_of.isoformat(),
        eligible_company_count=len(company_ids),
        notes=(reason,),
    )


def select_quarterly_companies(
    universe: pd.DataFrame,
    current_tickers: Iterable[str],
) -> SelectionResult:
    """Apply the current quarterly rank, replacement and fast-entry rules."""
    data = universe.copy()
    current_tickers = {str(value).upper().strip() for value in current_tickers}
    data["ticker"] = data["ticker"].astype("string").str.upper().str.strip()
    data["is_current"] = data["ticker"].isin(current_tickers)
    for column, default in [
        ("regular_eligible", False),
        ("fast_entry_eligible", False),
        ("company_full_market_cap", np.nan),
    ]:
        if column not in data:
            data[column] = default

    company = (
        data.sort_values("company_full_market_cap", ascending=False, na_position="last")
        .groupby("company_id", as_index=False, sort=False)
        .agg(
            company_full_market_cap=("company_full_market_cap", "max"),
            is_current=("is_current", "max"),
            regular_eligible=("regular_eligible", "max"),
            fast_entry_eligible=("fast_entry_eligible", "max"),
        )
        .sort_values("company_full_market_cap", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    ranked = company.loc[company["regular_eligible"] | company["is_current"]].copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    current_companies = set(ranked.loc[ranked["is_current"], "company_id"])
    selected = set(current_companies)

    removals = ranked.loc[
        ranked["is_current"] & ranked["rank"].gt(125), "company_id"
    ].tolist()
    for company_id in reversed(removals):
        selected.discard(company_id)

    target_count = len(current_companies)
    replacements = ranked.loc[
        ranked["regular_eligible"] & ~ranked["company_id"].isin(selected)
    ]
    for company_id in replacements["company_id"]:
        if len(selected) >= target_count:
            break
        selected.add(company_id)

    current_ranked = ranked.loc[ranked["company_id"].isin(current_companies)]
    top_40_threshold = (
        float(current_ranked.iloc[min(39, len(current_ranked) - 1)]["company_full_market_cap"])
        if not current_ranked.empty
        else np.inf
    )
    fast_entries = company.loc[
        company["fast_entry_eligible"]
        & ~company["company_id"].isin(selected)
        & company["company_full_market_cap"].ge(top_40_threshold)
    ]
    selected.update(fast_entries["company_id"])

    selected_rows = data.loc[data["company_id"].isin(selected)].copy()
    # Existing companies retain their currently represented securities. New
    # companies bring all otherwise eligible listed classes into the simulation.
    selected_rows["selected"] = selected_rows["is_current"] | selected_rows.get(
        "base_eligible", True
    )
    selected_rows = selected_rows.loc[selected_rows["selected"]].copy()
    selected_tickers = tuple(selected_rows["ticker"].drop_duplicates())
    selected_current = set(selected_tickers).intersection(current_tickers)
    additions = tuple(sorted(set(selected_tickers).difference(current_tickers)))
    removals_tickers = tuple(sorted(current_tickers.difference(selected_current)))

    data["selected"] = data["ticker"].isin(selected_tickers)
    securities = data.loc[
        data["is_current"] | data["selected"],
        [
            "ticker",
            "company_name",
            "company_id",
            "security_type",
            "is_current",
            "selected",
        ],
    ].drop_duplicates("ticker", keep="first")
    return SelectionResult(
        securities=securities,
        selected_tickers=selected_tickers,
        selected_company_ids=tuple(sorted(selected)),
        additions=additions,
        removals=removals_tickers,
        status="calculated",
        source="provided_universe",
        as_of=date.today().isoformat(),
        eligible_company_count=int(ranked["regular_eligible"].sum()),
        notes=(),
    )


def select_annual_companies(
    ranked_companies: pd.DataFrame,
    *,
    previous_constituents: Iterable[str],
    previous_top_100: Iterable[str],
    added_since_reconstitution: Iterable[str] = (),
) -> tuple[str, ...]:
    """Apply the annual top-75/100/125 company selection sequence."""
    ranked = ranked_companies.sort_values("full_market_cap", ascending=False).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    previous = set(previous_constituents)
    protected = set(previous_top_100).union(added_since_reconstitution)
    selected: list[str] = []

    def add(values: Iterable[str]) -> None:
        for value in values:
            if value not in selected and len(selected) < 100:
                selected.append(value)

    add(ranked.loc[ranked["rank"].le(75), "company_id"])
    add(ranked.loc[ranked["company_id"].isin(previous) & ranked["rank"].le(100), "company_id"])
    add(
        ranked.loc[
            ranked["company_id"].isin(previous.intersection(protected))
            & ranked["rank"].between(101, 125),
            "company_id",
        ]
    )
    add(ranked.loc[ranked["rank"].le(100), "company_id"])
    return tuple(selected)


def simulate_rebalance(
    current_holdings: pd.DataFrame,
    reference_data: pd.DataFrame,
    selection: SelectionResult,
    *,
    rebalance_type: str = "quarterly",
) -> RebalanceResult:
    """Calculate post-review weights and their WDI against the same reference."""
    if rebalance_type not in {"quarterly", "annual"}:
        raise ValueError("rebalance_type must be quarterly or annual.")
    current = current_holdings[["ticker", "company_name", "actual_weight"]].copy()
    current["ticker"] = current["ticker"].astype("string").str.upper().str.strip()
    current["actual_weight"] = pd.to_numeric(current["actual_weight"], errors="coerce")
    current = current.loc[current["actual_weight"].fillna(0).gt(0)]
    current["actual_weight"] /= current["actual_weight"].sum()

    securities = selection.securities.copy()
    securities["ticker"] = securities["ticker"].astype("string").str.upper().str.strip()
    data = securities.merge(current, on="ticker", how="outer", suffixes=("", "_current"))
    data["company_name"] = data["company_name"].fillna(data["company_name_current"])
    data["company_id"] = data["company_id"].fillna(
        data["ticker"].map(lambda value: f"TICKER:{value}")
    )
    data["selected"] = data["selected"].fillna(
        data["ticker"].isin(selection.selected_tickers)
    )
    data["is_current"] = data["is_current"].fillna(
        data["ticker"].isin(set(current["ticker"]))
    )
    data["actual_weight"] = data["actual_weight"].fillna(0.0)
    data = data.merge(
        reference_data,
        on="ticker",
        how="left",
        suffixes=("", "_reference"),
    )
    if "security_type_reference" in data:
        data["security_type"] = data["security_type"].fillna(
            data["security_type_reference"]
        )
    for column in [
        "reference_weight_raw",
        "modified_float_mass_raw",
        "counterfactual_reference_raw",
        "float_shares",
        "shares_outstanding",
        "price",
    ]:
        if column not in data:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["modified_float_mass_raw"] = data["modified_float_mass_raw"].fillna(
        data["reference_weight_raw"]
    )
    data["counterfactual_reference_raw"] = data[
        "counterfactual_reference_raw"
    ].fillna(data["reference_weight_raw"])

    calibration = data.loc[
        data["is_current"]
        & data["actual_weight"].gt(0)
        & data["modified_float_mass_raw"].gt(0),
        "modified_float_mass_raw",
    ] / data.loc[
        data["is_current"]
        & data["actual_weight"].gt(0)
        & data["modified_float_mass_raw"].gt(0),
        "actual_weight",
    ]
    current_mass_fallback = (
        data["is_current"]
        & data["selected"]
        & data["actual_weight"].gt(0)
        & ~data["modified_float_mass_raw"].gt(0)
    )
    if not calibration.empty and current_mass_fallback.any():
        data.loc[current_mass_fallback, "modified_float_mass_raw"] = (
            data.loc[current_mass_fallback, "actual_weight"]
            * float(calibration.median())
        )
        data.loc[current_mass_fallback, "rebalance_input_status"] = (
            "current_weight_modified_cap_fallback"
        )

    data["listed_total_cap"] = data["price"] * data["shares_outstanding"]
    conversion_scale, conversion_count = derive_acwi_total_cap_conversion(data)
    direct_acwi = (
        data.get("reference_source", pd.Series(None, index=data.index))
        .astype("string")
        .eq("ishares_acwi")
        .fillna(False)
        & data["modified_float_mass_raw"].gt(0)
    )
    fallback_reference = ~direct_acwi & data["modified_float_mass_raw"].gt(0)
    yfinance_float_ratio = data["shares_outstanding"] / data["float_shares"]
    yfinance_ratio_valid = (
        yfinance_float_ratio.replace([np.inf, -np.inf], np.nan)
        .between(1.0, 10.0)
    )

    data["modified_cap_ratio"] = np.nan
    data["modified_market_cap_mass"] = np.nan
    data["rebalance_input_status"] = "missing_modified_cap_mass"

    if conversion_scale is not None:
        converted_total_mass = data["listed_total_cap"] * conversion_scale
        acwi_ratio = (
            converted_total_mass / data["modified_float_mass_raw"]
        ).clip(lower=1.0, upper=3.0)
        direct_valid = direct_acwi & converted_total_mass.gt(0)
        data.loc[direct_valid, "modified_cap_ratio"] = acwi_ratio.loc[direct_valid]
        data.loc[direct_valid, "modified_market_cap_mass"] = (
            data.loc[direct_valid, "modified_float_mass_raw"]
            * acwi_ratio.loc[direct_valid]
        )
        data.loc[direct_valid, "rebalance_input_status"] = (
            "valid_acwi_converted_tso"
        )
        direct_without_total = direct_acwi & ~direct_valid
    else:
        direct_without_total = direct_acwi

    data.loc[direct_without_total, "modified_cap_ratio"] = 1.0
    data.loc[direct_without_total, "modified_market_cap_mass"] = data.loc[
        direct_without_total, "modified_float_mass_raw"
    ]
    data.loc[direct_without_total, "rebalance_input_status"] = (
        "acwi_total_cap_conversion_unavailable_1x"
    )

    fallback_ratio = yfinance_float_ratio.clip(lower=1.0, upper=3.0).where(
        yfinance_ratio_valid,
        1.0,
    )
    data.loc[fallback_reference, "modified_cap_ratio"] = fallback_ratio.loc[
        fallback_reference
    ]
    data.loc[fallback_reference, "modified_market_cap_mass"] = (
        data.loc[fallback_reference, "modified_float_mass_raw"]
        * fallback_ratio.loc[fallback_reference]
    )
    data.loc[
        fallback_reference & yfinance_ratio_valid,
        "rebalance_input_status",
    ] = "valid_yfinance_float_fallback"
    data.loc[
        fallback_reference & ~yfinance_ratio_valid,
        "rebalance_input_status",
    ] = "float_ratio_fallback_1x"
    data.loc[
        current_mass_fallback,
        "rebalance_input_status",
    ] = "current_weight_modified_cap_fallback"

    weighting_valid = data["selected"] & data["modified_market_cap_mass"].gt(0)
    score_valid = weighting_valid & data["counterfactual_reference_raw"].gt(0)
    if not weighting_valid.any():
        raise ValueError("No selected security has a valid rebalance reference mass.")

    valid = data.loc[weighting_valid].copy()
    security_initial = valid.set_index("ticker")["modified_market_cap_mass"]
    security_initial = security_initial / security_initial.sum()
    company_initial = (
        valid.assign(initial_weight=valid["ticker"].map(security_initial))
        .groupby("company_id")["initial_weight"]
        .sum()
    )
    company_final = apply_company_capping(company_initial)
    company_scale = company_final / company_initial
    security_final = security_initial * valid.set_index("ticker")["company_id"].map(
        company_scale
    )
    security_final = security_final / security_final.sum()
    if rebalance_type == "annual":
        security_final = apply_annual_security_capping(security_final)

    score_tickers = data.loc[score_valid, "ticker"]
    score_weights = security_final.reindex(score_tickers).dropna()
    score_weights = score_weights / score_weights.sum()
    reference = data.loc[score_valid].set_index("ticker")[
        "counterfactual_reference_raw"
    ]
    reference = reference / reference.sum()
    data["rebalance_weight"] = data["ticker"].map(security_final).fillna(0.0)
    data["rebalance_reference_weight"] = data["ticker"].map(reference)
    data["rebalance_weight_change"] = (
        data["rebalance_weight"] - data["actual_weight"]
    )
    data["rebalance_weight_delta"] = data["ticker"].map(
        score_weights - reference
    )
    data["rebalance_distortion_contribution"] = (
        50.0 * data["rebalance_weight_delta"].abs()
    )
    data["rebalance_membership"] = data["selected"].astype(bool)

    current_selected_weight = float(
        data.loc[data["is_current"] & data["selected"], "actual_weight"].sum()
    )
    valid_current_selected_weight = float(
        data.loc[
            data["is_current"] & score_valid,
            "actual_weight",
        ].sum()
    )
    coverage = (
        valid_current_selected_weight / current_selected_weight
        if current_selected_weight > 0
        else 1.0
    )
    score = float(data.loc[score_valid, "rebalance_distortion_contribution"].sum())
    input_fallbacks = int(
        data.loc[weighting_valid, "rebalance_input_status"]
        .isin(
            [
                "float_ratio_fallback_1x",
                "current_weight_modified_cap_fallback",
                "acwi_total_cap_conversion_unavailable_1x",
            ]
        )
        .sum()
    )
    notes = list(selection.notes)
    if conversion_scale is not None:
        notes.append(
            "Direct ACWI modified caps use an ACWI/total-cap conversion scale "
            f"of {conversion_scale:.8g}, calibrated on {conversion_count} "
            "matched securities; yfinance floatShares is not used for them."
        )
    if input_fallbacks:
        notes.append(
            f"{input_fallbacks} securities used a documented modified-cap fallback "
            "because a public float/TSO input was unavailable or inconsistent."
        )
    status = (
        selection.status
        if coverage >= 0.99
        else f"{selection.status}_partial_coverage"
    )
    return RebalanceResult(
        components=data,
        ndx_wdi=score,
        coverage_ratio=coverage,
        constituent_count=int(weighting_valid.sum()),
        status=status,
        method=f"{rebalance_type}_modified_market_cap_2026",
        reference_date=selection.as_of,
        additions=selection.additions,
        removals=selection.removals,
        data_source=selection.source,
        acwi_conversion_scale=conversion_scale,
        acwi_calibration_count=conversion_count,
        notes=tuple(notes),
    )


def derive_acwi_total_cap_conversion(
    data: pd.DataFrame,
    *,
    quantile: float = 0.90,
    minimum_pairs: int = 20,
) -> tuple[float | None, int]:
    """Estimate fund-value units per total-cap dollar from direct ACWI matches.

    ACWI holding market values are proportional to free-float capitalization.
    The upper quantile of ACWI mass / listed total capitalization estimates the
    common conversion scale from names whose free float is close to 100%.
    """
    if not 0.5 <= quantile < 1.0:
        raise ValueError("quantile must be between 0.5 and 1.0.")
    reference_source = data.get(
        "reference_source",
        pd.Series(None, index=data.index),
    ).astype("string")
    float_mass = pd.to_numeric(
        data.get("modified_float_mass_raw"), errors="coerce"
    )
    listed_total_cap = pd.to_numeric(
        data.get("listed_total_cap"), errors="coerce"
    )
    valid = (
        reference_source.eq("ishares_acwi")
        & float_mass.gt(0)
        & listed_total_cap.gt(0)
    )
    ratios = (float_mass.loc[valid] / listed_total_cap.loc[valid]).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(ratios) < minimum_pairs:
        return None, int(len(ratios))
    return float(ratios.quantile(quantile)), int(len(ratios))


def apply_company_capping(initial_weights: pd.Series) -> pd.Series:
    """Apply the 2026 company-level 20% and aggregate 40% adjustments."""
    weights = _normalize_positive(initial_weights)
    # The official constraints assume the approximately 100-company NDX
    # universe. Tiny synthetic fixtures cannot mathematically absorb a 20% cap.
    if len(weights) < 5:
        return weights
    base = weights.copy()
    capped: set[object] = set()
    for _ in range(len(weights) + 1):
        capped.update(weights.index[weights.gt(0.24 + 1e-12)])
        if not capped:
            break
        caps = pd.Series(1.0, index=weights.index)
        caps.loc[list(capped)] = 0.20
        weights = _proportional_with_caps(base, caps)
        if not weights.drop(index=list(capped)).gt(0.24 + 1e-12).any():
            break

    stage_two_base = weights.copy()
    large = set(weights.index[weights.gt(0.045 + 1e-12)])
    if float(weights.loc[list(large)].sum()) >= 0.48 - 1e-12:
        large_index = weights.index.isin(large)
        large_base = stage_two_base.loc[large_index]
        small_base = stage_two_base.loc[~large_index]
        if small_base.empty:
            raise ValueError(
                "Company-level aggregate capping requires companies outside "
                "the greater-than-4.5% cohort."
            )
        adjusted = pd.Series(0.0, index=weights.index)
        adjusted.loc[large_index] = 0.40 * large_base / large_base.sum()
        smallest_large = float(adjusted.loc[large_index].min())
        outside_cap = min(0.045, smallest_large)
        adjusted.loc[~large_index] = _proportional_with_caps(
            small_base / small_base.sum(),
            pd.Series(outside_cap / 0.60, index=small_base.index),
        ) * 0.60
        weights = adjusted
    return weights / weights.sum()


def apply_annual_security_capping(initial_weights: pd.Series) -> pd.Series:
    """Apply the annual security-level 14%, top-five and 4.4% constraints."""
    weights = _normalize_positive(initial_weights)
    base = weights.copy()
    if weights.gt(0.15 + 1e-12).any():
        caps = pd.Series(1.0, index=weights.index)
        caps.loc[weights.gt(0.15 + 1e-12)] = 0.14
        weights = _proportional_with_caps(base, caps)

    stage_two_base = weights.copy()
    top_five = list(weights.nlargest(min(5, len(weights))).index)
    if len(top_five) == 5 and float(weights.loc[top_five].sum()) >= 0.40 - 1e-12:
        top_base = stage_two_base.loc[top_five]
        outside = stage_two_base.drop(index=top_five)
        result = pd.Series(0.0, index=weights.index)
        result.loc[top_five] = 0.385 * top_base / top_base.sum()
        outside_cap = min(0.044, float(result.loc[top_five].min()))
        if not outside.empty:
            result.loc[outside.index] = _proportional_with_caps(
                outside / outside.sum(),
                pd.Series(outside_cap / 0.615, index=outside.index),
            ) * 0.615
        weights = result
    return weights / weights.sum()


def _normalize_positive(weights: pd.Series) -> pd.Series:
    result = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    result = result.loc[result.gt(0)]
    if result.empty:
        raise ValueError("At least one positive weight is required.")
    return result / result.sum()


def _proportional_with_caps(base: pd.Series, caps: pd.Series) -> pd.Series:
    """Solve weight_i=min(cap_i, adjustment_factor*base_i), summing to one."""
    base = _normalize_positive(base)
    caps = caps.reindex(base.index).fillna(1.0).astype(float)
    if float(caps.sum()) < 1.0 - 1e-12:
        raise ValueError("Capping constraints cannot sum to 100%.")
    low, high = 0.0, 1.0
    while float(np.minimum(caps, high * base).sum()) < 1.0:
        high *= 2.0
    for _ in range(100):
        middle = (low + high) / 2.0
        total = float(np.minimum(caps, middle * base).sum())
        if total < 1.0:
            low = middle
        else:
            high = middle
    result = np.minimum(caps, high * base)
    return result / result.sum()


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
