"""BlackRock ACWI reference weights and calibrated yfinance fallbacks."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from io import StringIO
from typing import Sequence

import numpy as np
import pandas as pd
import requests

from market_data_provider import FLOAT_SHARES_OVERRIDE_STATUS


DEFAULT_ACWI_URL = (
    "https://www.ishares.com/us/products/239600/"
    "ishares-msci-acwi-etf/latest-holdings.csv"
)
DEFAULT_ADR_TICKERS = frozenset({"ARM", "ASML", "PDD"})
ADR_NAME_PATTERN = re.compile(r"\b(?:ADR|ADS)\b|DEPOSITARY", re.IGNORECASE)

REFERENCE_COLUMNS = [
    "ticker",
    "reference_weight_raw",
    "reference_source",
    "security_type",
    "acwi_weight",
    "acwi_market_value",
    "acwi_listing",
    "reference_status",
]


def parse_acwi_holdings_csv(raw_csv: str) -> pd.DataFrame:
    """Parse the official iShares ACWI holdings download."""
    lines = raw_csv.lstrip("\ufeff").splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "Ticker" in line
            and "Name" in line
            and "Asset Class" in line
            and "Weight (%)" in line
        ),
        None,
    )
    if header_index is None:
        raise ValueError("The ACWI download does not contain a holdings header.")

    frame = pd.read_csv(
        StringIO("\n".join(lines[header_index:])),
        thousands=",",
        on_bad_lines="skip",
    )
    required = {
        "Ticker",
        "Name",
        "Asset Class",
        "Market Value",
        "Weight (%)",
        "Location",
        "Exchange",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing ACWI columns: {sorted(missing)}")

    frame = frame.loc[
        frame["Asset Class"].astype("string").str.casefold().eq("equity")
    ].copy()
    frame["ticker"] = frame["Ticker"].astype("string").str.upper().str.strip()
    frame["acwi_name"] = frame["Name"].astype("string").str.strip()
    frame["acwi_market_value"] = pd.to_numeric(
        frame["Market Value"], errors="coerce"
    )
    frame["acwi_weight"] = (
        pd.to_numeric(frame["Weight (%)"], errors="coerce") / 100.0
    )
    frame["acwi_location"] = frame["Location"].astype("string").str.strip()
    frame["acwi_exchange"] = frame["Exchange"].astype("string").str.strip()
    frame = frame.loc[
        frame["ticker"].notna()
        & frame["acwi_market_value"].notna()
        & (frame["acwi_market_value"] > 0)
    ].copy()

    as_of_match = re.search(
        r'Fund Holdings as of,\s*"?(?P<date>[^"\r\n]+)', raw_csv
    )
    frame.attrs["holdings_as_of"] = (
        as_of_match.group("date").strip() if as_of_match else None
    )
    frame.attrs["published_weight_total"] = float(frame["acwi_weight"].sum())
    return frame[
        [
            "ticker",
            "acwi_name",
            "acwi_market_value",
            "acwi_weight",
            "acwi_location",
            "acwi_exchange",
        ]
    ].reset_index(drop=True)


@dataclass
class IsharesAcwiFloatWeightsProvider:
    """Build free-float reference masses from the public ACWI portfolio."""

    url: str = DEFAULT_ACWI_URL
    timeout: int = 30
    min_equity_rows: int = 1_500
    min_calibration_rows: int = 20
    source_name: str = "ishares_acwi_public_holdings"
    holdings_as_of: str | None = None

    def get_holdings(self) -> pd.DataFrame:
        response = requests.get(
            self.url,
            timeout=self.timeout,
            headers={"User-Agent": "NDX-WDI/1.0"},
        )
        response.raise_for_status()
        frame = parse_acwi_holdings_csv(response.text)
        if len(frame) < self.min_equity_rows:
            raise ValueError(
                f"ACWI returned {len(frame)} equity rows; "
                f"at least {self.min_equity_rows} are required."
            )
        weight_total = float(frame.attrs.get("published_weight_total", 0))
        if not 0.90 <= weight_total <= 1.01:
            raise ValueError(
                f"ACWI published equity weights total {weight_total:.2%}; "
                "expected 90% to 101%."
            )
        self.holdings_as_of = frame.attrs.get("holdings_as_of")
        return frame

    def build_reference(
        self,
        holdings: pd.DataFrame,
        market_data: pd.DataFrame,
        *,
        adr_tickers: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Return ACWI masses plus calibrated yfinance fallback masses."""
        acwi = self.get_holdings()
        references = match_acwi_holdings(
            holdings,
            acwi,
            adr_tickers=adr_tickers,
        )
        return add_yfinance_fallbacks(
            references,
            market_data,
            min_calibration_rows=self.min_calibration_rows,
        )


def configured_adr_tickers() -> set[str]:
    configured = {
        ticker.strip().upper()
        for ticker in os.getenv("NDX_ADR_TICKERS", "").split(",")
        if ticker.strip()
    }
    return set(DEFAULT_ADR_TICKERS).union(configured)


def classify_security_types(holdings: pd.DataFrame) -> pd.DataFrame:
    """Label ADR/ADS rows from names plus the maintained ticker override."""
    known_adrs = configured_adr_tickers()
    rows = []
    for holding in holdings.itertuples(index=False):
        ticker = str(holding.ticker).upper().strip()
        company_name = str(holding.company_name)
        is_adr = ticker in known_adrs or bool(ADR_NAME_PATTERN.search(company_name))
        rows.append(
            {
                "ticker": ticker,
                "security_type": "ADR/ADS" if is_adr else "Ordinary share",
            }
        )
    return pd.DataFrame(rows).drop_duplicates("ticker", keep="last")


def match_acwi_holdings(
    holdings: pd.DataFrame,
    acwi: pd.DataFrame,
    *,
    adr_tickers: Sequence[str] | None = None,
    minimum_name_score: float = 0.55,
) -> pd.DataFrame:
    """Match by ticker and company name while excluding ADR/ADS securities."""
    required_holdings = {"ticker", "company_name"}
    missing = required_holdings.difference(holdings.columns)
    if missing:
        raise ValueError(f"Missing holdings columns: {sorted(missing)}")
    required_acwi = {
        "ticker",
        "acwi_name",
        "acwi_market_value",
        "acwi_weight",
        "acwi_location",
        "acwi_exchange",
    }
    missing = required_acwi.difference(acwi.columns)
    if missing:
        raise ValueError(f"Missing ACWI columns: {sorted(missing)}")

    known_adrs = configured_adr_tickers()
    if adr_tickers is not None:
        known_adrs.update(str(ticker).upper() for ticker in adr_tickers)

    rows: list[dict[str, object]] = []
    acwi_by_ticker = {
        str(ticker): group
        for ticker, group in acwi.groupby("ticker", sort=False)
    }
    for holding in holdings.itertuples(index=False):
        ticker = str(holding.ticker).upper().strip()
        company_name = str(holding.company_name)
        is_adr = ticker in known_adrs or bool(ADR_NAME_PATTERN.search(company_name))
        row: dict[str, object] = {
            "ticker": ticker,
            "reference_weight_raw": np.nan,
            "reference_source": "yfinance_fallback" if is_adr else None,
            "security_type": "ADR/ADS" if is_adr else "Ordinary share",
            "acwi_weight": np.nan,
            "acwi_market_value": np.nan,
            "acwi_listing": None,
            "reference_status": "pending_yfinance_fallback",
        }
        candidates = acwi_by_ticker.get(ticker)
        if not is_adr and candidates is not None:
            scored = candidates.copy()
            scored["_name_score"] = [
                _name_similarity(company_name, name)
                for name in scored["acwi_name"]
            ]
            best = scored.sort_values(
                ["_name_score", "acwi_market_value"],
                ascending=[False, False],
            ).iloc[0]
            if float(best["_name_score"]) >= minimum_name_score:
                row.update(
                    {
                        "reference_weight_raw": float(best["acwi_market_value"]),
                        "reference_source": "ishares_acwi",
                        "acwi_weight": float(best["acwi_weight"]),
                        "acwi_market_value": float(best["acwi_market_value"]),
                        "acwi_listing": (
                            f"{best['acwi_location']} / {best['acwi_exchange']}"
                        ),
                        "reference_status": "valid_acwi",
                    }
                )
        rows.append(row)
    return pd.DataFrame(rows, columns=REFERENCE_COLUMNS)


def add_yfinance_fallbacks(
    references: pd.DataFrame,
    market_data: pd.DataFrame,
    *,
    min_calibration_rows: int = 20,
) -> pd.DataFrame:
    """Convert valid yfinance float caps into ACWI fund-value units."""
    data = references.merge(market_data, on="ticker", how="left")
    for column in ["price", "float_shares", "shares_outstanding", "market_cap"]:
        if column not in data:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")

    price_valid = data["price"].notna() & (data["price"] > 0)
    float_valid = data["float_shares"].notna() & (data["float_shares"] > 0)
    outstanding_valid = (
        data["shares_outstanding"].notna() & (data["shares_outstanding"] > 0)
    )
    market_cap_valid = data["market_cap"].notna() & (data["market_cap"] > 0)
    float_share_status = data.get(
        "float_shares_status",
        pd.Series(None, index=data.index),
    ).astype("string")
    hardcoded_override = float_share_status.eq(
        FLOAT_SHARES_OVERRIDE_STATUS
    ).fillna(False)
    float_cap = data["price"] * data["float_shares"]
    inconsistent = (
        (
            float_valid
            & outstanding_valid
            & (data["float_shares"] > data["shares_outstanding"] * 1.10)
        )
        | (
            price_valid
            & float_valid
            & market_cap_valid
            & (float_cap > data["market_cap"] * 1.25)
        )
    )
    yfinance_valid = price_valid & float_valid & ~inconsistent

    direct = data["reference_source"].eq("ishares_acwi")
    calibration = direct & yfinance_valid
    fallback = ~direct
    if fallback.any():
        calibration_ratios = (
            data.loc[calibration, "acwi_market_value"]
            / float_cap.loc[calibration]
        )
        calibration_ratios = calibration_ratios.replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if len(calibration_ratios) < min_calibration_rows:
            raise ValueError(
                "Only "
                f"{len(calibration_ratios)} ACWI/yfinance pairs are available; "
                f"{min_calibration_rows} are required to calibrate fallbacks."
            )
        calibration_scale = float(calibration_ratios.median())
        valid_fallback = fallback & yfinance_valid
        data.loc[valid_fallback, "reference_weight_raw"] = (
            float_cap.loc[valid_fallback] * calibration_scale
        )
        data.loc[valid_fallback, "reference_source"] = "yfinance_fallback"
        data.loc[valid_fallback, "reference_status"] = "valid_yfinance_fallback"
        valid_override = valid_fallback & hardcoded_override
        data.loc[valid_override, "reference_source"] = (
            "hardcoded_float_override"
        )
        data.loc[valid_override, "reference_status"] = (
            "valid_hardcoded_float_override"
        )
        data.loc[fallback & inconsistent, "reference_status"] = (
            "invalid_yfinance_fallback"
        )
        data.loc[
            fallback & ~inconsistent & ~price_valid & float_valid,
            "reference_status",
        ] = "missing_price_yfinance_fallback"
        data.loc[
            fallback & ~inconsistent & ~float_valid,
            "reference_status",
        ] = "missing_float_yfinance_fallback"
        data.attrs["yfinance_fallback_scale"] = calibration_scale
        data.attrs["calibration_count"] = len(calibration_ratios)

    result = data[REFERENCE_COLUMNS].copy()
    result.attrs.update(data.attrs)
    return result


def _name_similarity(left: object, right: object) -> float:
    return SequenceMatcher(None, _normalize_name(left), _normalize_name(right)).ratio()


def _normalize_name(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()
