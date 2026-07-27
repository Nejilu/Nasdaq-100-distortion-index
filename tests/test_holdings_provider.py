import math
import warnings

import pandas as pd

import qqq_holdings_provider as holdings_module
from qqq_holdings_provider import (
    HoldingsProviderChain,
    IsharesSpreadsheetXmlHoldingsProvider,
    parse_qqq_holdings_csv,
)
from snapshot_service import build_holdings_chain, build_spx_holdings_chain


def test_holdings_parser_removes_non_equities_and_normalizes():
    raw = """Fund,QQQ
As of,2026-07-17
Ticker,Name,Asset Class,Weight (%)
A,Alpha,Equity,50
B,Beta,Equity,30
C,Gamma,Equity,19
USD,US Dollar,Cash,1
"""
    holdings = parse_qqq_holdings_csv(raw)

    assert holdings["ticker"].tolist() == ["A", "B", "C"]
    assert math.isclose(holdings["actual_weight"].sum(), 1.0)
    assert math.isclose(holdings.set_index("ticker").loc["A", "actual_weight"], 50 / 99)


def test_decimal_weights_are_not_divided_by_100():
    raw = """Ticker,Name,Asset Class,Weight
A,Alpha,Equity,0.6
B,Beta,Equity,0.4
"""
    holdings = parse_qqq_holdings_csv(raw)
    assert holdings.set_index("ticker").loc["A", "actual_weight"] == 0.6


class _FrameProvider:
    def __init__(self, name, fund, count):
        self.source_name = name
        self.reference_fund = fund
        self.count = count

    def get_holdings(self):
        return pd.DataFrame(
            {
                "ticker": [f"T{index:03d}" for index in range(self.count)],
                "company_name": [f"Company {index}" for index in range(self.count)],
                "actual_weight": [1 / self.count] * self.count,
            }
        )


def test_provider_chain_rejects_top10_and_uses_complete_fallback():
    chain = HoldingsProviderChain(
        [_FrameProvider("nasdaq_top10", "NDX", 10), _FrameProvider("iqq_full", "IQQ", 103)]
    )
    holdings = chain.get_holdings()

    assert len(holdings) == 103
    assert chain.source_name == "iqq_full"
    assert chain.reference_fund == "IQQ"
    assert "Unexpected constituent count" in chain.failures[0]


def test_provider_chain_rejects_incomplete_published_weight_total():
    partial = _FrameProvider("partial", "QNDX", 100).get_holdings()
    partial.attrs["published_weight_total"] = 0.82

    class PartialProvider:
        source_name = "partial"
        reference_fund = "QNDX"

        @staticmethod
        def get_holdings():
            return partial

    chain = HoldingsProviderChain([PartialProvider(), _FrameProvider("cndx_full", "CNDX", 101)])
    holdings = chain.get_holdings()

    assert len(holdings) == 101
    assert chain.reference_fund == "CNDX"
    assert "82.00%" in chain.failures[0]


def test_ishares_spreadsheet_xml_parser(monkeypatch):
    rows = "".join(
        f"<ss:Row><ss:Cell><ss:Data>{ticker}</ss:Data></ss:Cell>"
        f"<ss:Cell><ss:Data>{name}</ss:Data></ss:Cell>"
        "<ss:Cell><ss:Data>EQUITY</ss:Data></ss:Cell>"
        "<ss:Cell><ss:Data>Technology</ss:Data></ss:Cell>"
        "<ss:Cell><ss:Data>Equity</ss:Data></ss:Cell>"
        "<ss:Cell><ss:Data>100</ss:Data></ss:Cell>"
        f"<ss:Cell><ss:Data>{weight}</ss:Data></ss:Cell></ss:Row>"
        for ticker, name, weight in [("A", "Alpha", 50), ("B", "Beta", 30), ("C", "Gamma", 20)]
    )
    xml = f"""<ss:Workbook><ss:Worksheet ss:Name="Holdings"><ss:Table>
    <ss:Row><ss:Cell><ss:Data>Fund Holdings as of</ss:Data></ss:Cell><ss:Cell><ss:Data>Jul 17, 2026</ss:Data></ss:Cell></ss:Row>
    <ss:Row><ss:Cell><ss:Data>Ticker</ss:Data></ss:Cell><ss:Cell><ss:Data>Name</ss:Data></ss:Cell><ss:Cell><ss:Data>Type</ss:Data></ss:Cell><ss:Cell><ss:Data>Sector</ss:Data></ss:Cell><ss:Cell><ss:Data>Asset Class</ss:Data></ss:Cell><ss:Cell><ss:Data>Market Value</ss:Data></ss:Cell><ss:Cell><ss:Data>Weight (%)</ss:Data></ss:Cell></ss:Row>
    {rows}</ss:Table></ss:Worksheet></ss:Workbook>"""

    class Response:
        text = xml

        @staticmethod
        def raise_for_status():
            return None

    monkeypatch.setattr(holdings_module.requests, "get", lambda *args, **kwargs: Response())
    filters_before = list(warnings.filters)
    holdings = IsharesSpreadsheetXmlHoldingsProvider().get_holdings()

    assert holdings["ticker"].tolist() == ["A", "B", "C"]
    assert math.isclose(holdings["actual_weight"].sum(), 1.0)
    assert holdings.attrs["holdings_as_of"] == "Jul 17, 2026"
    assert warnings.filters == filters_before


def test_ishares_funds_are_prioritized_before_invesco(monkeypatch):
    for variable in (
        "NON_UCITS_HOLDINGS_CSV",
        "UCITS_HOLDINGS_CSV",
        "QQQ_HOLDINGS_CSV",
    ):
        monkeypatch.delenv(variable, raising=False)

    non_ucits = build_holdings_chain("non_ucits")
    ucits = build_holdings_chain("ucits")

    assert [provider.reference_fund for provider in non_ucits.providers[:2]] == [
        "IQQ",
        "QQQ",
    ]
    assert [provider.reference_fund for provider in ucits.providers[:2]] == [
        "CNDX",
        "EQQQ",
    ]


def test_spx_chains_use_matching_ishares_funds(monkeypatch):
    monkeypatch.delenv("NON_UCITS_SPX_HOLDINGS_CSV", raising=False)
    monkeypatch.delenv("UCITS_SPX_HOLDINGS_CSV", raising=False)

    non_ucits = build_spx_holdings_chain("non_ucits")
    ucits = build_spx_holdings_chain("ucits")

    assert non_ucits.min_constituents == 450
    assert ucits.max_constituents == 550
    assert [provider.reference_fund for provider in non_ucits.providers] == ["IVV"]
    assert [provider.reference_fund for provider in ucits.providers] == ["CSPX"]
