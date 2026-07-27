"""Quarterly NDX distortion history derived from QQQ and SPGM N-PORT filings."""

from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from hashlib import sha256
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import requests


EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
SEC_ARCHIVE_TEMPLATE = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/primary_doc.xml"
)
DEFAULT_READER_PROXY = "https://r.jina.ai/http://www.sec.gov/Archives/edgar/data"
DEFAULT_HISTORY_PATH = "data/edgar_quarterly_history.csv"
DEFAULT_ARCHIVE_DIR = "data/sec_nport_filings"
QQQ_CIK = "0001067839"
SPGM_CIK = "0001168164"
SPGM_SERIES_ID = "S000036082"

FLATTENED_POSITION_PATTERN = re.compile(
    r"(?<![A-Z0-9])"
    r"(?P<cusip>[A-Z0-9]{9})\s+"
    r"(?P<balance>-?\d+(?:\.\d+)?)\s+"
    r"(?P<units>[A-Z]{2})\s+"
    r"(?P<currency>[A-Z]{3})\s+"
    r"(?P<value>-?\d+(?:\.\d+)?)\s+"
    r"(?P<pct>-?\d+(?:\.\d+)?)\s+"
    r"(?P<direction>Long|Short)\s+"
    r"(?P<asset>[A-Z]{2})\s+"
)

HISTORY_COLUMNS = [
    "report_date",
    "ndx_wdi",
    "ndx_wdi_raw",
    "correction_points",
    "correction_method",
    "correction_ratio_median",
    "coverage_ratio",
    "qqq_equity_count",
    "spgm_equity_count",
    "matched_count",
    "estimated_count",
    "excluded_qqq_count",
    "excluded_non_comparable_count",
    "qqq_matched_weight",
    "spgm_matched_weight",
    "rebalance_type",
    "qqq_accession",
    "spgm_accession",
    "qqq_filed_date",
    "spgm_filed_date",
]

NON_COMPARABLE_NAME_PATTERN = re.compile(
    r"\bADR\b|DEPOSITARY|NEW YORK SHARES",
    flags=re.IGNORECASE,
)
CORRECTION_METHOD = "median_observed_overweight_ratio"


@dataclass(frozen=True)
class NportFiling:
    fund: str
    cik: str
    report_date: str
    filed_date: str
    accession: str

    @property
    def archive_url(self) -> str:
        return SEC_ARCHIVE_TEMPLATE.format(
            cik=str(int(self.cik)),
            accession=self.accession.replace("-", ""),
        )


def discover_quarterly_filings(
    *,
    timeout: int = 30,
) -> tuple[list[NportFiling], list[NportFiling]]:
    """Discover public quarterly QQQ and SPGM N-PORT filings in EDGAR."""
    qqq = _search_filings(
        fund="QQQ",
        cik=QQQ_CIK,
        query="",
        timeout=timeout,
    )
    spgm = _search_filings(
        fund="SPGM",
        cik=SPGM_CIK,
        query=SPGM_SERIES_ID,
        timeout=timeout,
    )
    return qqq, spgm


def _search_filings(
    *,
    fund: str,
    cik: str,
    query: str,
    timeout: int,
) -> list[NportFiling]:
    response = requests.get(
        EDGAR_SEARCH_URL,
        params={
            "q": query,
            "ciks": cik,
            "forms": "NPORT-P",
            "from": 0,
            "size": 100,
        },
        headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "gzip, deflate"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    filings: dict[str, NportFiling] = {}
    for hit in payload.get("hits", {}).get("hits", []):
        if not str(hit.get("_id", "")).endswith(":primary_doc.xml"):
            continue
        source = hit.get("_source", {})
        report_date = str(source.get("period_ending") or "")
        accession = str(source.get("adsh") or "")
        if not report_date or not accession:
            continue
        filings[report_date] = NportFiling(
            fund=fund,
            cik=cik,
            report_date=report_date,
            filed_date=str(source.get("file_date") or ""),
            accession=accession,
        )
    result = sorted(filings.values(), key=lambda filing: filing.report_date)
    if not result:
        raise ValueError(f"No public N-PORT filings were found for {fund}.")
    return result


def parse_nport_positions(content: str) -> pd.DataFrame:
    """Parse equity CUSIPs and reported portfolio percentages."""
    stripped = content.lstrip()
    if stripped.startswith("<") and "edgarsubmission" in stripped[:1_000].casefold():
        positions = _parse_nport_xml(content)
    else:
        positions = _parse_flattened_nport(content)
    if positions.empty:
        raise ValueError("No equity positions with CUSIPs were parsed from N-PORT.")
    positions["cusip"] = positions["cusip"].astype("string").str.upper().str.strip()
    for column in ["pct_value", "value_usd", "balance"]:
        if column not in positions:
            positions[column] = np.nan
        positions[column] = pd.to_numeric(positions[column], errors="coerce")
    if "security_name" not in positions:
        positions["security_name"] = None
    positions = positions.loc[
        positions["cusip"].str.fullmatch(r"[A-Z0-9]{9}", na=False)
        & ~positions["cusip"].eq("000000000")
        & positions["pct_value"].notna()
        & (positions["pct_value"] > 0)
    ]
    return positions.groupby("cusip", as_index=False, sort=False).agg(
        security_name=("security_name", "first"),
        pct_value=("pct_value", "sum"),
        value_usd=("value_usd", "sum"),
        balance=("balance", "sum"),
        units=("units", "first"),
        currency=("currency", "first"),
    )


def _parse_nport_xml(content: str) -> pd.DataFrame:
    sanitized = re.sub(
        r"&(?!#\d+;|#x[0-9A-Fa-f]+;|[A-Za-z][A-Za-z0-9]+;)",
        "&amp;",
        content,
    )
    root = ET.fromstring(sanitized)
    rows = []
    for investment in root.iter():
        if _local_name(investment.tag).casefold() != "invstorsec":
            continue
        values = {
            _local_name(element.tag).casefold(): (element.text or "").strip()
            for element in investment.iter()
        }
        if (
            values.get("assetcat") == "EC"
            and values.get("payoffprofile", "Long") == "Long"
            and values.get("cusip")
            and values.get("pctval")
        ):
            rows.append(
                {
                    "cusip": values["cusip"],
                    "security_name": values.get("title") or values.get("name"),
                    "pct_value": values["pctval"],
                    "value_usd": values.get("valusd"),
                    "balance": values.get("balance"),
                    "units": values.get("units"),
                    "currency": values.get("curcd"),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "cusip",
            "security_name",
            "pct_value",
            "value_usd",
            "balance",
            "units",
            "currency",
        ],
    )


def _parse_flattened_nport(content: str) -> pd.DataFrame:
    rows = []
    for match in FLATTENED_POSITION_PATTERN.finditer(content):
        if match.group("asset") != "EC" or match.group("direction") != "Long":
            continue
        rows.append(
            {
                "cusip": match.group("cusip"),
                "security_name": _flattened_security_name(content, match.start()),
                "pct_value": match.group("pct"),
                "value_usd": match.group("value"),
                "balance": match.group("balance"),
                "units": match.group("units"),
                "currency": match.group("currency"),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "cusip",
            "security_name",
            "pct_value",
            "value_usd",
            "balance",
            "units",
            "currency",
        ],
    )


def calculate_quarterly_point(
    qqq: pd.DataFrame,
    spgm: pd.DataFrame,
) -> dict[str, float | int]:
    """Calculate raw and missing-weight-corrected quarterly scores."""
    components = calculate_quarterly_components(qqq, spgm)
    return _summarize_quarterly_components(
        components,
        qqq_equity_count=len(qqq),
        spgm_equity_count=len(spgm),
    )


def _summarize_quarterly_components(
    components: pd.DataFrame,
    *,
    qqq_equity_count: int,
    spgm_equity_count: int,
) -> dict[str, float | int]:
    """Summarize a previously calculated quarterly component table."""
    matched = components["match_status"].eq("matched")
    estimated = components["correction_status"].eq("estimated_missing_spgm")
    non_comparable = components["correction_status"].eq(
        "excluded_non_comparable"
    )
    qqq_total = float(
        components.loc[~non_comparable, "qqq_raw_weight"].sum()
    )
    qqq_matched_total = float(
        components.loc[matched, "qqq_raw_weight"].sum()
    )
    spgm_matched_total = float(
        components.loc[matched, "spgm_raw_weight"].sum()
    )
    raw_score = float(
        components.loc[matched, "distortion_contribution"].sum()
    )
    corrected_score = float(
        components["corrected_distortion_contribution"].sum()
    )
    return {
        "ndx_wdi": corrected_score,
        "ndx_wdi_raw": raw_score,
        "correction_points": corrected_score - raw_score,
        "correction_method": CORRECTION_METHOD,
        "correction_ratio_median": float(
            components["correction_ratio_median"].dropna().iloc[0]
        ),
        "coverage_ratio": qqq_matched_total / qqq_total,
        "qqq_equity_count": int(qqq_equity_count),
        "spgm_equity_count": int(spgm_equity_count),
        "matched_count": int(matched.sum()),
        "estimated_count": int(estimated.sum()),
        "excluded_qqq_count": int(
            components["match_status"].eq("excluded_not_in_spgm").sum()
        ),
        "excluded_non_comparable_count": int(non_comparable.sum()),
        "qqq_matched_weight": qqq_matched_total / 100.0,
        "spgm_matched_weight": spgm_matched_total / 100.0,
    }


def calculate_quarterly_components(
    qqq: pd.DataFrame,
    spgm: pd.DataFrame,
) -> pd.DataFrame:
    """Return raw matched weights and corrected weights for every QQQ security."""
    qqq_data = qqq.drop_duplicates("cusip").set_index("cusip")
    spgm_data = spgm.drop_duplicates("cusip").set_index("cusip")
    qqq_weights = qqq_data["pct_value"]
    spgm_weights = spgm_data["pct_value"]
    matched_cusips = qqq_weights.index.intersection(spgm_weights.index)
    matched_coverage = float(
        qqq_weights.loc[matched_cusips].sum() / qqq_weights.sum()
    )
    if len(matched_cusips) < 40 or matched_coverage < 0.80:
        raise ValueError(
            f"Only {len(matched_cusips)} QQQ equity CUSIPs representing "
            f"{matched_coverage:.2%} of QQQ were found in SPGM."
        )

    qqq_matched = qqq_weights.loc[matched_cusips]
    spgm_matched = spgm_weights.loc[matched_cusips]
    qqq_matched_total = float(qqq_matched.sum())
    spgm_matched_total = float(spgm_matched.sum())
    if min(qqq_matched_total, spgm_matched_total) <= 0:
        raise ValueError("Quarterly matched weight totals must be positive.")

    actual = qqq_matched / qqq_matched_total
    counterfactual = spgm_matched / spgm_matched_total
    matched = pd.DataFrame(
        {
            "cusip": matched_cusips,
            "security_name": qqq_data.loc[matched_cusips, "security_name"].values,
            "qqq_raw_weight": qqq_matched.values,
            "spgm_raw_weight": spgm_matched.values,
            "qqq_weight": actual.values,
            "spgm_weight": counterfactual.values,
            "match_status": "matched",
        }
    )
    matched["weight_delta"] = matched["qqq_weight"] - matched["spgm_weight"]
    matched["distortion_contribution"] = 50.0 * matched["weight_delta"].abs()

    excluded_cusips = qqq_weights.index.difference(matched_cusips)
    excluded = pd.DataFrame(
        {
            "cusip": excluded_cusips,
            "security_name": qqq_data.loc[excluded_cusips, "security_name"].values,
            "qqq_raw_weight": qqq_weights.loc[excluded_cusips].values,
            "spgm_raw_weight": np.nan,
            "qqq_weight": np.nan,
            "spgm_weight": np.nan,
            "match_status": "excluded_not_in_spgm",
            "weight_delta": np.nan,
            "distortion_contribution": np.nan,
        }
    )
    components = pd.concat([matched, excluded], ignore_index=True)
    return apply_missing_weight_correction(components)


def apply_missing_weight_correction(components: pd.DataFrame) -> pd.DataFrame:
    """Estimate absent comparable SPGM weights from the observed overweight median."""
    data = components.copy()
    matched = data["match_status"].eq("matched")
    names = data["security_name"].fillna("").astype(str)
    non_comparable = ~matched & names.str.contains(
        NON_COMPARABLE_NAME_PATTERN,
        regex=True,
    )
    estimated = ~matched & ~non_comparable
    correction_universe = ~non_comparable

    qqq_total = float(data.loc[correction_universe, "qqq_raw_weight"].sum())
    spgm_matched_total = float(data.loc[matched, "spgm_raw_weight"].sum())
    if min(qqq_total, spgm_matched_total) <= 0:
        raise ValueError("Corrected quarterly weight totals must be positive.")

    actual = data.loc[correction_universe, "qqq_raw_weight"] / qqq_total
    observed_reference = (
        data.loc[matched, "spgm_raw_weight"] / spgm_matched_total
    )
    matched_actual_total = float(actual.loc[matched].sum())
    observed_actual = actual.loc[matched] / matched_actual_total
    observed_ratio = observed_actual / observed_reference
    overrepresented = observed_ratio > 1.0
    if int(overrepresented.sum()) < 5:
        raise ValueError(
            "At least five observed overweights are required for correction."
        )

    # These ratios use the observed SPGM mass as their unit. This lets estimated
    # weights be appended to the observed SPGM distribution before final scaling.
    scaled_ratios = actual.loc[matched] / observed_reference
    correction_ratio = float(scaled_ratios.loc[overrepresented].median())
    if not np.isfinite(correction_ratio) or correction_ratio <= 0:
        raise ValueError("The missing-weight correction ratio is invalid.")

    reference_units = pd.Series(np.nan, index=data.index, dtype=float)
    reference_units.loc[matched] = observed_reference
    reference_units.loc[estimated] = (
        actual.loc[estimated] / correction_ratio
    )
    corrected_reference = reference_units / float(reference_units.sum())

    data["corrected_qqq_weight"] = np.nan
    data.loc[correction_universe, "corrected_qqq_weight"] = actual
    data["corrected_spgm_weight"] = corrected_reference
    data["corrected_weight_delta"] = (
        data["corrected_qqq_weight"] - data["corrected_spgm_weight"]
    )
    data["corrected_distortion_contribution"] = (
        50.0 * data["corrected_weight_delta"].abs()
    )
    data["correction_status"] = "observed_spgm"
    data.loc[estimated, "correction_status"] = "estimated_missing_spgm"
    data.loc[non_comparable, "correction_status"] = "excluded_non_comparable"
    data["correction_ratio_median"] = correction_ratio
    return data


def build_quarterly_history(
    *,
    archive_dir: str | Path = DEFAULT_ARCHIVE_DIR,
    max_workers: int = 4,
    timeout: int = 60,
) -> pd.DataFrame:
    """Download, match, and calculate every shared quarterly filing date."""
    qqq_filings, spgm_filings = discover_quarterly_filings(timeout=timeout)
    qqq_by_date = {filing.report_date: filing for filing in qqq_filings}
    spgm_by_date = {filing.report_date: filing for filing in spgm_filings}
    shared_dates = sorted(set(qqq_by_date).intersection(spgm_by_date))
    if len(shared_dates) < 20:
        raise ValueError(
            f"Only {len(shared_dates)} shared QQQ/SPGM quarters were discovered."
        )

    filings = [
        filing
        for report_date in shared_dates
        for filing in (qqq_by_date[report_date], spgm_by_date[report_date])
    ]
    archive_path = Path(archive_dir)
    positions_by_accession = _load_positions(
        filings,
        archive_dir=archive_path,
        max_workers=max_workers,
        timeout=timeout,
    )
    rows = []
    all_positions = []
    all_components = []
    for report_date in shared_dates:
        qqq_filing = qqq_by_date[report_date]
        spgm_filing = spgm_by_date[report_date]
        qqq_positions = positions_by_accession[qqq_filing.accession]
        spgm_positions = positions_by_accession[spgm_filing.accession]
        components = calculate_quarterly_components(qqq_positions, spgm_positions)
        point = _summarize_quarterly_components(
            components,
            qqq_equity_count=len(qqq_positions),
            spgm_equity_count=len(spgm_positions),
        )
        components.insert(0, "report_date", report_date)
        all_components.append(components)
        for filing, positions in [
            (qqq_filing, qqq_positions),
            (spgm_filing, spgm_positions),
        ]:
            position_rows = positions.copy()
            position_rows.insert(0, "accession", filing.accession)
            position_rows.insert(0, "report_date", report_date)
            position_rows.insert(0, "fund", filing.fund)
            all_positions.append(position_rows)
        rows.append(
            {
                "report_date": report_date,
                **point,
                "rebalance_type": _rebalance_type(report_date),
                "qqq_accession": qqq_filing.accession,
                "spgm_accession": spgm_filing.accession,
                "qqq_filed_date": qqq_filing.filed_date,
                "spgm_filed_date": spgm_filing.filed_date,
            }
        )
    history = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    archive_path.mkdir(parents=True, exist_ok=True)
    pd.concat(all_positions, ignore_index=True).to_csv(
        archive_path / "positions.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.concat(all_components, ignore_index=True).to_csv(
        archive_path / "constituent_history.csv.gz",
        index=False,
        compression="gzip",
    )
    _write_manifest(filings, archive_path)
    return history


def save_quarterly_history(
    frame: pd.DataFrame,
    path: str | Path = DEFAULT_HISTORY_PATH,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, float_format="%.12g")
    return destination


def _load_positions(
    filings: Sequence[NportFiling],
    *,
    archive_dir: Path,
    max_workers: int,
    timeout: int,
) -> dict[str, pd.DataFrame]:
    worker_count = max(1, min(int(max_workers), 6))
    result: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _get_filing_positions,
                filing,
                archive_dir=archive_dir,
                timeout=timeout,
            ): filing
            for filing in filings
        }
        for future in as_completed(futures):
            filing = futures[future]
            result[filing.accession] = future.result()
            print(
                f"Archived {filing.fund} {filing.report_date} "
                f"({len(result)}/{len(filings)})",
                flush=True,
            )
    return result


def _get_filing_positions(
    filing: NportFiling,
    *,
    archive_dir: Path,
    timeout: int,
) -> pd.DataFrame:
    filing_dir = (
        archive_dir
        / "raw"
        / filing.fund.lower()
        / filing.report_date
        / filing.accession
    )
    xml_path = filing_dir / "primary_doc.xml"
    text_path = filing_dir / "primary_doc.txt"
    if xml_path.exists():
        content = xml_path.read_text(encoding="utf-8")
    else:
        content, transport = _download_filing(filing, timeout=timeout)
        filing_dir.mkdir(parents=True, exist_ok=True)
        destination = xml_path if content.lstrip().startswith("<") else text_path
        destination.write_text(content, encoding="utf-8")
        (filing_dir / "transport.txt").write_text(transport, encoding="ascii")
    return parse_nport_positions(content)


def _download_filing(filing: NportFiling, *, timeout: int) -> tuple[str, str]:
    direct = requests.get(
        filing.archive_url,
        headers=_sec_headers(),
        timeout=timeout,
    )
    if direct.ok and _is_edgar_submission(direct.text):
        return direct.text, "sec_direct"

    proxy_base = os.getenv("SEC_READER_PROXY", DEFAULT_READER_PROXY).rstrip("/")
    proxy_url = (
        f"{proxy_base}/{int(filing.cik)}/"
        f"{filing.accession.replace('-', '')}/primary_doc.xml"
    )
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            response = requests.get(
                proxy_url,
                headers={
                    "X-Respond-With": "html",
                    "X-Engine": "curl",
                    "X-Timeout": str(min(timeout, 180)),
                },
                timeout=max(timeout, 180),
            )
            if response.status_code == 429:
                last_error = RuntimeError(
                    f"Reader rate limit for {filing.accession}"
                )
                retry_after = int(response.headers.get("Retry-After", "0") or 0)
                time.sleep(max(retry_after, min(2 ** (attempt + 1), 30)))
                continue
            response.raise_for_status()
            if not _is_edgar_submission(response.text):
                raise ValueError(
                    "The SEC reader fallback did not return a complete N-PORT XML root."
                )
            return response.text, "sec_reader_xml"
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < 5:
                time.sleep(2 ** attempt)
    raise RuntimeError(
        f"Unable to retrieve SEC filing {filing.accession}: {last_error}"
    )


def _is_edgar_submission(content: str) -> bool:
    """Return whether the response begins with an EDGAR submission XML root."""
    prefix = content[:1_000].lstrip("\ufeff \t\r\n")
    if prefix.casefold().startswith("<?xml"):
        declaration_end = prefix.find("?>")
        if declaration_end < 0:
            return False
        prefix = prefix[declaration_end + 2 :].lstrip()
    return bool(re.match(r"<\s*edgarsubmission\b", prefix, flags=re.IGNORECASE))


def _sec_headers() -> dict[str, str]:
    return {
        "User-Agent": os.getenv(
            "SEC_USER_AGENT",
            "NDX-WDI/0.2 https://github.com/Nejilu/NDX-distortion-index",
        ),
        "Accept-Encoding": "gzip, deflate",
    }


def _rebalance_type(report_date: str) -> str:
    date = pd.Timestamp(report_date)
    if date.year == 2023 and date.month == 9:
        return "special_rebalance"
    if date.month == 12:
        return "annual_reconstitution"
    return "quarterly_rebalance"


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _flattened_security_name(content: str, cusip_start: int) -> str | None:
    current_line_start = content.rfind("\n", 0, cusip_start) + 1
    previous_line_end = max(0, current_line_start - 1)
    previous_line_start = content.rfind("\n", 0, previous_line_end) + 1
    previous_line = content[previous_line_start:previous_line_end].strip()
    identifiers = list(re.finditer(r"[A-Z0-9]{20}", previous_line))
    if not identifiers:
        return None
    title = previous_line[identifiers[-1].end() :].strip()
    return title or None


def _write_manifest(filings: Sequence[NportFiling], archive_dir: Path) -> None:
    rows = []
    for filing in filings:
        filing_dir = (
            archive_dir
            / "raw"
            / filing.fund.lower()
            / filing.report_date
            / filing.accession
        )
        local_path = next(
            (
                path
                for path in [
                    filing_dir / "primary_doc.xml",
                    filing_dir / "primary_doc.txt",
                ]
                if path.exists()
            ),
            None,
        )
        if local_path is None:
            raise FileNotFoundError(f"No local archive exists for {filing.accession}.")
        content = local_path.read_bytes()
        transport_path = filing_dir / "transport.txt"
        transport = (
            transport_path.read_text(encoding="ascii").strip()
            if transport_path.exists()
            else (
                "sec_direct"
                if local_path.suffix.casefold() == ".xml"
                else "sec_reader_fallback"
            )
        )
        rows.append(
            {
                "fund": filing.fund,
                "report_date": filing.report_date,
                "filed_date": filing.filed_date,
                "accession": filing.accession,
                "sec_url": filing.archive_url,
                "local_path": local_path.relative_to(archive_dir).as_posix(),
                "transport": transport,
                "sha256": sha256(content).hexdigest(),
                "bytes": len(content),
            }
        )
    pd.DataFrame(rows).sort_values(["report_date", "fund"]).to_csv(
        archive_dir / "manifest.csv",
        index=False,
    )
