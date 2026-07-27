"""QQQ holdings providers.

The public provider is deliberately isolated behind a small protocol so it can be
replaced by a licensed or official index feed without changing the calculation.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Protocol, Sequence

import pandas as pd
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning


DEFAULT_INVESCO_URL = (
    "https://www.invesco.com/us/financial-products/etfs/holdings/main/holdings/0"
    "?audienceType=Investor&action=download&ticker=QQQ"
)
DEFAULT_IQQ_URL = (
    "https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/api/v1/"
    "get-fund-document?appSubType=ISHARES&appType=PRODUCT_PAGE&component=fundDownload&"
    "locale=en_US&portfolioId=351653&targetSite=us-ishares&userType=individual"
)
DEFAULT_CNDX_URL = (
    "https://www.ishares.com/uk/individual/en/products/253741/"
    "ishares-nasdaq-100-ucits-etf-acc-fund/1506575576011.ajax?"
    "fileType=csv&fileName=CSNDX_holdings&dataType=fund"
)
DEFAULT_EQQQ_URL = (
    "https://www.invesco.com/uk/financial-products/etfs/holdings/main/holdings/0"
    "?audienceType=Investor&action=download&ticker=EQQQ"
)
DEFAULT_IVV_URL = (
    "https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf/"
    "latest-holdings.csv"
)
DEFAULT_CSPX_URL = (
    "https://www.ishares.com/uk/individual/en/products/253743/"
    "ishares-core-sp-500-ucits-etf/1506575576011.ajax?"
    "fileType=csv&fileName=CSPX_holdings&dataType=fund"
)


class HoldingsProvider(Protocol):
    """Contract implemented by holdings data sources."""

    source_name: str
    reference_fund: str

    def get_holdings(self) -> pd.DataFrame:
        """Return ticker, company_name and normalized actual_weight columns."""


@dataclass
class InvescoQQQHoldingsProvider:
    """Read the public Invesco QQQ holdings export or a compatible local CSV."""

    url: str = DEFAULT_INVESCO_URL
    csv_path: str | Path | None = None
    timeout: int = 30
    source_name: str = "invesco_qqq_public_holdings"
    reference_fund: str = "QQQ"

    def get_holdings(self) -> pd.DataFrame:
        if self.csv_path:
            raw = Path(self.csv_path).read_text(encoding="utf-8-sig")
        else:
            response = requests.get(
                self.url,
                timeout=self.timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 NDX-WDI-MVP/1.0",
                    "Accept": "text/csv,application/csv,application/octet-stream,*/*;q=0.8",
                    "Referer": "https://www.invesco.com/qqq-etf/en/about.html",
                },
            )
            if response.status_code == 406:
                raise RuntimeError(
                    "Invesco is currently rejecting automated downloads (HTTP 406). "
                    "Use QQQ_HOLDINGS_CSV or --holdings-csv."
                )
            response.raise_for_status()
            raw = response.text
        return parse_qqq_holdings_csv(raw)


@dataclass
class CsvHoldingsProvider:
    """Local provider for an explicitly supplied issuer holdings CSV."""

    csv_path: str | Path
    source_name: str = "local_holdings_csv"
    reference_fund: str = "local_csv"

    def get_holdings(self) -> pd.DataFrame:
        raw = Path(self.csv_path).read_text(encoding="utf-8-sig")
        return parse_qqq_holdings_csv(raw)


@dataclass
class HttpCsvHoldingsProvider:
    """Generic provider for issuer-published CSV exports."""

    url: str
    source_name: str
    reference_fund: str
    timeout: int = 30

    def get_holdings(self) -> pd.DataFrame:
        response = requests.get(
            self.url,
            timeout=self.timeout,
            headers={
                "User-Agent": "Mozilla/5.0 NDX-WDI-MVP/1.0",
                "Accept": "text/csv,application/csv,application/octet-stream,*/*;q=0.8",
            },
        )
        response.raise_for_status()
        if response.content.lstrip().lower().startswith(b"<!doctype html"):
            raise ValueError("The source returned an HTML page instead of a holdings export.")
        raw = response.content.decode("utf-8-sig", errors="replace")
        frame = parse_qqq_holdings_csv(raw)
        frame.attrs["holdings_as_of"] = _extract_as_of(raw)
        return frame


@dataclass
class IsharesSpreadsheetXmlHoldingsProvider:
    """Parse the official BlackRock SpreadsheetML data download used by IQQ."""

    url: str = DEFAULT_IQQ_URL
    timeout: int = 30
    source_name: str = "blackrock_ishares_iqq_data_download"
    reference_fund: str = "IQQ"

    def get_holdings(self) -> pd.DataFrame:
        response = requests.get(
            self.url,
            timeout=self.timeout,
            headers={"User-Agent": "Mozilla/5.0 NDX-WDI-MVP/1.0"},
        )
        response.raise_for_status()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
            document = BeautifulSoup(response.text, "html.parser")
        worksheet = next(
            (
                sheet
                for sheet in document.find_all("ss:worksheet")
                if str(sheet.get("ss:name", "")).lower() == "holdings"
            ),
            None,
        )
        if worksheet is None:
            raise ValueError("The IQQ download does not contain a Holdings worksheet.")

        rows = [
            [cell.get_text(" ", strip=True) for cell in row.find_all("ss:data")]
            for row in worksheet.find_all("ss:row")
        ]
        header_index = next(
            (
                index
                for index, row in enumerate(rows)
                if "Ticker" in row and any(value in row for value in {"Weight (%)", "Weight"})
            ),
            None,
        )
        if header_index is None:
            raise ValueError("The IQQ holdings header could not be found.")
        header = rows[header_index]
        records = [row for row in rows[header_index + 1 :] if len(row) == len(header)]
        frame = parse_qqq_holdings_csv(pd.DataFrame(records, columns=header).to_csv(index=False))

        as_of = None
        for row in rows[:header_index]:
            if row and row[0] == "Fund Holdings as of" and len(row) > 1:
                as_of = row[1]
                break
        frame.attrs["holdings_as_of"] = as_of
        return frame


@dataclass
class HoldingsProviderChain:
    """Use the first complete published holdings source in priority order."""

    providers: Sequence[HoldingsProvider]
    min_constituents: int = 90
    max_constituents: int = 130
    source_name: str = "unresolved"
    reference_fund: str = "unresolved"
    holdings_as_of: str | None = None
    failures: tuple[str, ...] = ()

    def get_holdings(self) -> pd.DataFrame:
        failures: list[str] = []
        for provider in self.providers:
            try:
                holdings = provider.get_holdings()
                validate_published_holdings(
                    holdings,
                    min_constituents=self.min_constituents,
                    max_constituents=self.max_constituents,
                )
                self.source_name = provider.source_name
                self.reference_fund = provider.reference_fund
                self.holdings_as_of = holdings.attrs.get("holdings_as_of")
                self.failures = tuple(failures)
                return holdings
            except Exception as exc:
                failures.append(f"{provider.source_name}: {type(exc).__name__}: {exc}")
        self.failures = tuple(failures)
        raise RuntimeError("No complete holdings source is available. " + " | ".join(failures))


def validate_published_holdings(
    holdings: pd.DataFrame,
    *,
    min_constituents: int = 90,
    max_constituents: int = 130,
) -> None:
    """Reject partial/top-10 extracts and malformed published weights."""
    required = {"ticker", "company_name", "actual_weight"}
    if not required.issubset(holdings.columns):
        raise ValueError(f"Missing columns: {sorted(required.difference(holdings.columns))}")
    count = len(holdings)
    if count < min_constituents or count > max_constituents:
        raise ValueError(
            f"Unexpected constituent count ({count}); expected between "
            f"{min_constituents} and {max_constituents}."
        )
    if holdings["ticker"].duplicated().any():
        raise ValueError("Duplicate tickers remain after normalization.")
    published_total = holdings.attrs.get("published_weight_total")
    if published_total is not None and not 0.95 <= float(published_total) <= 1.05:
        raise ValueError(
            f"Published weights cover only {float(published_total):.2%} of the fund."
        )
    total = float(holdings["actual_weight"].sum())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Published weights sum to {total}, not 1.")


def _extract_as_of(raw: str) -> str | None:
    for line in raw.splitlines()[:10]:
        if "as of" in line.lower():
            match = re.search(r"as of[^,]*,?[\"']?([^\"']+)[\"']?", line, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip(" ,\"")
    return None


def _canonical(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _find_header_row(raw: str) -> int:
    """Find a CSV header even when the issuer prepends a title/date block."""
    for index, line in enumerate(raw.splitlines()[:30]):
        cells = {_canonical(cell) for cell in line.split(",")}
        if "ticker" in cells and any(cell in cells for cell in {"weight", "weightpercent", "weightpct"}):
            return index
    return 0


def _select_column(frame: pd.DataFrame, candidates: set[str], required: bool = True) -> str | None:
    columns = {_canonical(column): column for column in frame.columns}
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    if required:
        raise ValueError(
            f"None of the columns {sorted(candidates)} were found. "
            f"Received columns: {list(frame.columns)}"
        )
    return None


def parse_qqq_holdings_csv(raw: str) -> pd.DataFrame:
    """Parse, filter and normalize an Invesco-like holdings CSV.

    Cash, futures, options and other non-equity rows are excluded. GOOG and
    GOOGL remain separate because grouping is performed by ticker only.
    """
    frame = pd.read_csv(StringIO(raw), skiprows=_find_header_row(raw))
    if frame.empty:
        raise ValueError("The QQQ holdings file is empty.")

    ticker_col = _select_column(frame, {"ticker", "holdingsticker", "symbol"})
    name_col = _select_column(frame, {"name", "holdingname", "companyname", "description"}, required=False)
    weight_col = _select_column(frame, {"weight", "weightpercent", "weightpct", "portfoliopercent"})
    asset_col = _select_column(frame, {"assetclass", "assettype", "securitytype", "classofshares"}, required=False)

    result = pd.DataFrame(
        {
            "ticker": frame[ticker_col].astype("string").str.strip().str.upper(),
            "company_name": (
                frame[name_col].astype("string").str.strip()
                if name_col
                else frame[ticker_col].astype("string").str.strip()
            ),
            "raw_weight": pd.to_numeric(
                frame[weight_col].astype("string").str.replace("%", "", regex=False).str.replace(",", "", regex=False),
                errors="coerce",
            ),
        }
    )

    if asset_col:
        asset = frame[asset_col].astype("string").str.lower()
        explicit_non_equity = asset.str.contains(
            r"cash|currency|future|option|swap|bond|fixed income|money market|derivative",
            regex=True,
            na=False,
        )
        result = result.loc[~explicit_non_equity].copy()

    non_security_ticker = result["ticker"].fillna("").str.contains(
        r"^(?:|--|-|N/A|NA|CASH|USD|US DOLLAR)$|CASH|FUTURE|OPTION",
        regex=True,
        na=True,
    )
    result = result.loc[~non_security_ticker].copy()
    result = result.loc[result["raw_weight"].notna() & (result["raw_weight"] > 0)].copy()
    if result.empty:
        raise ValueError("The QQQ holdings contain no valid equity positions.")

    # Issuer exports normally use percentages (e.g. 8.7). Compatible test files
    # may already use decimals. Detect the scale without silently altering rows.
    if result["raw_weight"].sum() > 1.5:
        result["raw_weight"] = result["raw_weight"] / 100.0

    result = (
        result.groupby("ticker", as_index=False, sort=False)
        .agg(company_name=("company_name", "first"), raw_weight=("raw_weight", "sum"))
    )
    total = float(result["raw_weight"].sum())
    if total <= 0:
        raise ValueError("The sum of QQQ equity weights must be positive.")
    result["actual_weight"] = result["raw_weight"] / total
    normalized = result[["ticker", "company_name", "actual_weight"]].reset_index(drop=True)
    normalized.attrs["published_weight_total"] = total
    return normalized
