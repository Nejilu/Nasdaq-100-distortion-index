"""Streamlit dashboard for non-UCITS and UCITS NDX distortion snapshots."""

from __future__ import annotations

import html
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from database import SnapshotDatabase
from rebalance_analytics import (
    AnnualRebalanceAnalysis,
    analyze_annual_rebalance,
)
from snapshot_service import recompute_snapshot


load_dotenv()
st.set_page_config(
    page_title="Nasdaq-100 Analytics",
    page_icon=":material/analytics:",
    layout="wide",
)

if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False


def _toggle_dark_mode() -> None:
    st.session_state["dark_mode"] = not bool(st.session_state["dark_mode"])


IS_DARK_MODE = bool(st.session_state["dark_mode"])
THEME = (
    {
        "bg": "#101519",
        "surface": "#171e23",
        "ink": "#edf2f5",
        "muted": "#9cabb5",
        "line": "#303b43",
        "soft": "#202a30",
        "chart_font": "#cdd7dd",
        "chart_grid": "#2a343b",
        "chart_zero": "#5d6a73",
        "marker_outline": "#101519",
        "note_ink": "#f0c36c",
        "note_bg": "#2a2418",
        "note_line": "#5e4b26",
    }
    if IS_DARK_MODE
    else {
        "bg": "#f7f8fa",
        "surface": "#ffffff",
        "ink": "#1b2733",
        "muted": "#687582",
        "line": "#dce2e7",
        "soft": "#eef2f4",
        "chart_font": "#33414d",
        "chart_grid": "#e7ebee",
        "chart_zero": "#aab4bd",
        "marker_outline": "#ffffff",
        "note_ink": "#76561f",
        "note_bg": "#fff6df",
        "note_line": "#efd8a5",
    }
)

NDX_ACTIVE_COLOR = "#df6b4f"
SPX_ACTIVE_COLOR = "#2f5f98"
ACTIVE_OVERLAP_COLOR = "#76558f"

dashboard_css = """
    <style>
    :root {
        --ndx-bg: __BG__;
        --ndx-surface: __SURFACE__;
        --ndx-ink: __INK__;
        --ndx-muted: __MUTED__;
        --ndx-line: __LINE__;
        --ndx-soft: __SOFT__;
        --ndx-teal: #177e78;
        --ndx-blue: #3f7dc0;
        --ndx-coral: #d45a57;
        --ndx-amber: #d69a2d;
        --ndx-green: #268463;
    }

    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: var(--ndx-bg) !important;
        color: var(--ndx-ink) !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    .block-container {
        max-width: 1480px;
        padding-top: 1.35rem;
        padding-bottom: 2.5rem;
    }

    h1, h2, h3 {
        color: var(--ndx-ink);
        letter-spacing: 0;
    }

    h1 {
        font-size: 2rem !important;
        line-height: 1.15 !important;
        margin-bottom: 0.15rem !important;
    }

    h3 {
        font-size: 1.05rem !important;
        line-height: 1.3 !important;
        margin-top: 0.2rem !important;
        margin-bottom: 0.1rem !important;
    }

    [data-testid="stCaptionContainer"] p {
        color: var(--ndx-muted);
        font-size: 0.82rem;
        line-height: 1.45;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stExpander"] summary,
    [data-testid="stTabs"] button {
        color: var(--ndx-ink);
    }

    [data-testid="stSegmentedControl"] button,
    [data-testid="stButton"] button,
    [data-testid="stPopover"] button {
        border-radius: 6px !important;
    }

    [data-testid="stBaseButton-secondary"],
    [data-testid="stPopoverButton"],
    button[role="radio"][aria-checked="false"] {
        color: var(--ndx-ink) !important;
        background: var(--ndx-surface) !important;
        border-color: var(--ndx-line) !important;
    }

    [data-testid="stBaseButton-secondary"] *,
    [data-testid="stPopoverButton"] *,
    button[role="radio"][aria-checked="false"] * {
        color: var(--ndx-ink) !important;
    }

    [data-testid="stButton"] button {
        min-height: 2.35rem;
    }

    [data-testid="stAlert"] {
        border-radius: 6px;
        padding: 0.55rem 0.75rem;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--ndx-line);
        border-radius: 6px;
        overflow: hidden;
        background: #ffffff;
        filter: __DATAFRAME_FILTER__;
        color-scheme: __COLOR_SCHEME__;
    }

    [data-testid="stExpander"] {
        border-color: var(--ndx-line);
        border-radius: 6px;
        background: var(--ndx-surface);
    }

    .ndx-score-strip {
        display: grid;
        grid-template-columns:
            minmax(250px, 1.5fr) minmax(180px, 1fr)
            repeat(4, minmax(85px, 0.62fr));
        align-items: center;
        gap: 0;
        margin: 0.55rem 0 0.35rem;
        padding: 0.8rem 0;
        border-top: 1px solid var(--ndx-line);
        border-bottom: 1px solid var(--ndx-line);
    }

    .ndx-score-main {
        padding: 0.1rem 1.2rem 0.1rem 0;
    }

    .ndx-eyebrow {
        color: var(--ndx-teal);
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .ndx-score-value {
        color: var(--ndx-ink);
        font-size: 3rem;
        font-weight: 680;
        line-height: 1;
        margin: 0.12rem 0 0.2rem;
    }

    .ndx-score-reading {
        display: flex;
        align-items: flex-end;
        gap: 0.9rem;
    }

    .ndx-score-meaning {
        max-width: 250px;
        padding-bottom: 0.24rem;
        color: var(--ndx-muted);
        font-size: 0.76rem;
        line-height: 1.35;
    }

    .ndx-score-meaning strong {
        color: var(--ndx-ink);
        font-weight: 650;
    }

    .ndx-score-context {
        color: var(--ndx-muted);
        font-size: 0.78rem;
    }

    .ndx-stat {
        min-height: 3.2rem;
        padding: 0.15rem 0.9rem;
        border-left: 1px solid var(--ndx-line);
    }

    .ndx-rebalance-score {
        min-height: 4.5rem;
        padding: 0.1rem 1rem;
        border-left: 1px solid var(--ndx-line);
    }

    .ndx-rebalance-value {
        color: var(--ndx-ink);
        font-size: 2.15rem;
        font-weight: 680;
        line-height: 1.15;
        margin-top: 0.12rem;
    }

    .ndx-rebalance-change {
        color: var(--ndx-muted);
        font-size: 0.72rem;
        line-height: 1.35;
    }

    .ndx-stat-label {
        color: var(--ndx-muted);
        font-size: 0.68rem;
        line-height: 1.2;
        text-transform: uppercase;
    }

    .ndx-stat-value {
        color: var(--ndx-ink);
        font-size: 1.05rem;
        font-weight: 620;
        line-height: 1.5;
    }

    .ndx-source-row {
        color: var(--ndx-muted);
        font-size: 0.78rem;
        line-height: 2.2rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .ndx-status-dot {
        display: inline-block;
        width: 0.48rem;
        height: 0.48rem;
        margin-right: 0.4rem;
        border-radius: 50%;
        background: var(--ndx-green);
        box-shadow: 0 0 0 3px rgba(38, 132, 99, 0.12);
    }

    .ndx-status-dot.partial {
        background: var(--ndx-amber);
        box-shadow: 0 0 0 3px rgba(214, 154, 45, 0.12);
    }

    .ndx-inline-note {
        display: inline-block;
        color: __NOTE_INK__;
        background: __NOTE_BG__;
        border: 1px solid __NOTE_LINE__;
        border-radius: 999px;
        padding: 0.2rem 0.55rem;
        font-size: 0.72rem;
        line-height: 1.25;
    }

    .ndx-section-rule {
        height: 1px;
        margin: 0.8rem 0 1rem;
        background: var(--ndx-line);
    }

    .ndx-active-strip {
        display: grid;
        grid-template-columns:
            minmax(260px, 1.3fr) minmax(230px, 1.1fr)
            repeat(2, minmax(150px, 0.72fr));
        align-items: center;
        margin: 0.55rem 0 0.45rem;
        padding: 0.8rem 0;
        border-top: 1px solid var(--ndx-line);
        border-bottom: 1px solid var(--ndx-line);
    }

    .ndx-active-main {
        padding-right: 1.2rem;
    }

    .ndx-active-value {
        color: var(--ndx-ink);
        font-size: 3rem;
        font-weight: 680;
        line-height: 1;
        margin: 0.12rem 0 0.2rem;
    }

    .ndx-active-meaning {
        padding: 0 1.1rem;
        color: var(--ndx-muted);
        font-size: 0.78rem;
        line-height: 1.42;
        border-left: 1px solid var(--ndx-line);
    }

    .ndx-active-meaning strong {
        color: var(--ndx-ink);
    }

    @media (max-width: 900px) {
        .block-container {
            padding-top: 0.9rem;
        }

        .st-key-theme_toggle {
            display: flex;
            justify-content: flex-end;
        }

        .st-key-theme_toggle button {
            width: 2.5rem !important;
            min-width: 2.5rem !important;
            padding: 0.35rem !important;
        }

        .st-key-theme_toggle button p {
            display: none;
        }

        .ndx-score-strip {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .ndx-active-strip {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .ndx-active-main {
            grid-column: 1 / -1;
            padding-bottom: 0.8rem;
        }

        .ndx-active-meaning,
        .ndx-active-strip .ndx-stat {
            border-left: 0;
            border-top: 1px solid var(--ndx-line);
            padding: 0.65rem 0.2rem;
        }

        .ndx-score-main {
            grid-column: 1 / -1;
            padding-bottom: 0.8rem;
        }

        .ndx-rebalance-score,
        .ndx-stat {
            border-left: 0;
            border-top: 1px solid var(--ndx-line);
            padding: 0.6rem 0.2rem;
        }

        .ndx-score-value {
            font-size: 2.55rem;
        }

        .ndx-score-meaning {
            max-width: 210px;
            font-size: 0.72rem;
        }
    }
    </style>
    """
for token, value in {
    "__BG__": THEME["bg"],
    "__SURFACE__": THEME["surface"],
    "__INK__": THEME["ink"],
    "__MUTED__": THEME["muted"],
    "__LINE__": THEME["line"],
    "__SOFT__": THEME["soft"],
    "__NOTE_INK__": THEME["note_ink"],
    "__NOTE_BG__": THEME["note_bg"],
    "__NOTE_LINE__": THEME["note_line"],
    "__DATAFRAME_FILTER__": (
        "invert(0.88) hue-rotate(180deg)" if IS_DARK_MODE else "none"
    ),
    "__COLOR_SCHEME__": "dark" if IS_DARK_MODE else "light",
}.items():
    dashboard_css = dashboard_css.replace(token, value)
st.html(dashboard_css)


def _database() -> SnapshotDatabase:
    return SnapshotDatabase(os.getenv("NDX_DB_PATH", "data/ndx_wdi.sqlite3"))


def _percent(value: float | None) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value:.2%}"


def _number(value: object, significant_digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{significant_digits}g}"


def _display_date(value: object) -> str:
    if value is None or str(value).strip() == "":
        return "not published"
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return parsed.strftime("%b %d, %Y")


def _component_table(
    frame: pd.DataFrame,
    weighting_basis: str,
    *,
    rebalanced_view: bool = False,
) -> pd.DataFrame:
    columns = [
        "ticker",
        "company_name",
        "security_type",
        "actual_weight",
        "counterfactual_weight",
        "weight_delta",
        "weight_ratio",
        "distortion_contribution",
        "price",
        "reference_source",
        "data_status",
    ]
    if rebalanced_view:
        columns.insert(5, "rebalance_weight_change")
        columns.insert(3, "rebalance_membership")
    if weighting_basis == "total":
        columns.insert(-1, "reference_shares")
    else:
        columns[-1:-1] = ["acwi_weight", "acwi_listing"]
    result = frame.reindex(columns=columns).copy()
    for column in ["actual_weight", "counterfactual_weight", "weight_delta"]:
        result[column] = result[column].map(_percent)
    if rebalanced_view:
        result["rebalance_weight_change"] = result["rebalance_weight_change"].map(
            lambda value: (
                "n/a" if pd.isna(value) else f"{value:+.2%}"
            )
        )
        result["rebalance_membership"] = result["rebalance_membership"].map(
            {1: "Included", 0: "Removed", True: "Included", False: "Removed"}
        ).fillna("n/a")
    result["weight_ratio"] = result["weight_ratio"].map(
        lambda value: "n/a" if pd.isna(value) else f"{value:.2f}x"
    )
    result["distortion_contribution"] = result["distortion_contribution"].map(
        lambda value: "n/a" if pd.isna(value) else f"{value:.3f}"
    )
    result["price"] = result["price"].map(
        lambda value: "n/a" if pd.isna(value) else f"${value:,.2f}"
    )
    if weighting_basis == "total":
        result["reference_shares"] = result["reference_shares"].map(
            lambda value: "n/a" if pd.isna(value) else f"{value:,.0f}"
        )
        result = result.rename(
            columns={
                "counterfactual_weight": "total_cap_weight",
                "reference_shares": "shares_outstanding",
            }
        )
    else:
        result["acwi_weight"] = result["acwi_weight"].map(_percent)
        result = result.rename(columns={"counterfactual_weight": "float_weight"})
    if rebalanced_view:
        result = result.rename(
            columns={
                "actual_weight": "annual_reconstitution_weight",
                "rebalance_weight_change": "change_vs_current",
                "rebalance_membership": "annual_membership_status",
            }
        )
    return result


def _table_for_display(
    frame: pd.DataFrame,
    weighting_basis: str,
    *,
    rebalanced_view: bool,
):
    table = _component_table(
        frame,
        weighting_basis,
        rebalanced_view=rebalanced_view,
    )
    if not rebalanced_view or "change_vs_current" not in table:
        return table
    return table.style.map(
        lambda value: (
            "color: #268463; font-weight: 650"
            if str(value).startswith("+")
            else (
                "color: #d45a57; font-weight: 650"
                if str(value).startswith("-")
                else ""
            )
        ),
        subset=["change_vs_current"],
    )


def _render_method_help(weighting_basis: str) -> None:
    label = "Free-float method" if weighting_basis == "float" else "Total-cap method"
    with st.popover(label, icon=":material/help:", width="stretch"):
        if weighting_basis == "float":
            st.markdown(
                """
                **Primary reference:** official iShares ACWI holding market values.

                ACWI matches are selected for Nasdaq-100 constituents and normalized
                to 100%. ADR/ADS securities and ACWI absences use a calibrated
                yfinance fallback. Each valid fallback is converted into ACWI
                fund-value units with the median conversion ratio from matched names.

                Free-float adjustment is also used by major investable benchmarks such
                as the [S&P 500](https://www.spglobal.com/spdji/en/methodology/article/sp-us-indices-methodology/)
                and [MSCI World](https://www.msci.com/indexes/index/990100/msci-world-index).
                """
            )
        else:
            st.markdown(
                """
                **Counterfactual:** price multiplied by all shares outstanding.

                This includes strategic holdings that may not be readily tradable.
                The [Nasdaq Composite](https://www.nasdaq.com/newsroom/nasdaq-composite-vs-nasdaq-100-what-investors-should-know)
                is a useful reference for total-capitalization weighting.
                """
            )
        st.caption(
            "The official Nasdaq-100 uses modified market capitalization with float "
            "and concentration constraints. Neither comparison reproduces it exactly."
        )


def _render_active_share_help() -> None:
    with st.popover(
        "Active Share method",
        icon=":material/help:",
        width="stretch",
    ):
        st.markdown(
            """
            **Active Share** is half the sum of absolute security-weight
            differences across the complete union of both portfolios.

            The non-UCITS comparison uses **IQQ vs IVV**. The UCITS comparison
            uses **CNDX vs CSPX**. Published equity weights are normalized after
            cash and derivative rows are removed.

            The annual scenario replaces published NDX weights with the same
            full reconstitution simulation used in the distortion view. S&P 500
            weights remain unchanged.
            """
        )


def _render_annual_reconstitution_help() -> None:
    with st.popover(
        "Annual review method",
        icon=":material/help:",
        width="stretch",
    ):
        st.markdown(
            """
            The panel reconstructs the full annual December review from the
            persisted public-data simulation.

            Initial weights use **Modified Market Capitalization**. Listed TSO
            is limited to three times free-float shares. Company constraints
            are then applied before security-level constraints.

            Selection follows the official top-75/100/125 sequence, but public
            membership is only a proxy for Nasdaq's non-public prior-review
            flags and discretionary inputs.

            - [Nasdaq-100 methodology](https://indexes.nasdaqomx.com/docs/Methodology_NDX.pdf)
            - [Nasdaq weight calculations](https://indexes.nasdaqomx.com/docs/Nasdaq_Index_Weight_Calculations.pdf)
            """
        )


def _render_score_strip(
    snapshot: dict[str, object],
    components: pd.DataFrame,
    universe_label: str,
    basis_label: str,
) -> None:
    weighting_basis = str(snapshot.get("weighting_basis") or "float")
    score = float(snapshot["ndx_wdi"])
    rebalance_score = snapshot.get("rebalance_ndx_wdi")
    rebalance_available = rebalance_score is not None and not pd.isna(rebalance_score)
    rebalance_value = (
        f"{float(rebalance_score):.2f}" if rebalance_available else "n/a"
    )
    score_change = (
        float(rebalance_score) - score if rebalance_available else None
    )
    score_change_label = (
        f"{score_change:+.2f} points vs live"
        if score_change is not None
        else "Refresh to calculate"
    )
    valid_fallbacks = int(
        components["data_status"].astype("string").eq("valid_yfinance_fallback").sum()
    )
    if weighting_basis == "float":
        fourth_label = "Fallbacks"
        fourth_value = str(valid_fallbacks)
        fifth_label = "Reference gaps"
        fifth_value = str(snapshot["missing_reference_shares_count"])
    else:
        fourth_label = "Missing shares"
        fourth_value = str(snapshot["missing_reference_shares_count"])
        fifth_label = "Missing prices"
        fifth_value = str(snapshot["missing_price_count"])

    st.html(
        f"""
        <div class="ndx-score-strip">
          <div class="ndx-score-main">
            <div class="ndx-eyebrow">Live NDX_WDI</div>
            <div class="ndx-score-reading">
              <div class="ndx-score-value">{score:.2f}</div>
              <div class="ndx-score-meaning">
                <strong>{score:.2f}% of index weight</strong> would need to be
                reallocated to match the reference.
              </div>
            </div>
            <div class="ndx-score-context">
              {html.escape(universe_label)} | {html.escape(basis_label)}
            </div>
          </div>
          <div class="ndx-rebalance-score">
            <div class="ndx-eyebrow">If annual reconstitution ran today</div>
            <div class="ndx-rebalance-value">{rebalance_value}</div>
            <div class="ndx-rebalance-change">{score_change_label}</div>
          </div>
          <div class="ndx-stat">
            <div class="ndx-stat-label">Coverage</div>
            <div class="ndx-stat-value">{_percent(snapshot["coverage_ratio"])}</div>
          </div>
          <div class="ndx-stat">
            <div class="ndx-stat-label">Constituents</div>
            <div class="ndx-stat-value">{int(snapshot["constituent_count"])}</div>
          </div>
          <div class="ndx-stat">
            <div class="ndx-stat-label">{fourth_label}</div>
            <div class="ndx-stat-value">{fourth_value}</div>
          </div>
          <div class="ndx-stat">
            <div class="ndx-stat-label">{fifth_label}</div>
            <div class="ndx-stat-value">{fifth_value}</div>
          </div>
        </div>
        """
    )


def _parse_json_list(value: object) -> list[str]:
    if value is None or str(value).strip() == "":
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return [str(value)]
    return [str(item) for item in parsed]


def _render_rebalance_controls(snapshot: dict[str, object]) -> bool:
    available = snapshot.get("rebalance_ndx_wdi") is not None
    columns = st.columns([4.9, 1.1], gap="small", vertical_alignment="center")
    with columns[0]:
        show_rebalanced = st.toggle(
            "Show annual-reconstitution weights",
            value=False,
            disabled=not available,
            key=f"rebalance_view_{snapshot['snapshot_id']}",
            help=(
                "Replace current ETF weights with the simulated Nasdaq-100 "
                "weights throughout the constituent views."
            ),
        )
    with columns[1]:
        with st.popover(
            "Review method",
            icon=":material/help:",
            width="stretch",
        ):
            st.markdown(
                """
                **Today simulation:** a full annual Nasdaq-100 reconstitution
                using current prices and current public inputs, as if today were
                the November reference date.

                Companies are ranked from the public Nasdaq universe. The screen
                applies Nasdaq listing tier, non-financial classification, security
                type, three-month liquidity and seasoning or fast-entry rules.

                For direct ACWI matches, modified-capitalization inputs use:
                `min(converted listed total cap, 3 x ACWI free-float mass)`.
                The conversion is calibrated from ACWI market values and listed
                total capitalizations. Yahoo `floatShares` is used only when ACWI
                cannot provide a direct reference.

                Initial weights are fully rebuilt from modified market
                capitalization. Company constraints are then applied
                iteratively: 24%/20%, followed by the 4.5%-48% cohort reduced
                to 40%.

                Security constraints are applied next: securities above 15%
                are reduced to 14%; when the five largest securities reach 40%,
                their aggregate is reduced to 38.5%, and every security outside
                the top five is capped at the lower of 4.4% or the fifth
                security's weight.
                """
            )
            additions = _parse_json_list(snapshot.get("rebalance_additions"))
            removals = _parse_json_list(snapshot.get("rebalance_removals"))
            st.markdown(
                f"""
                - **Reference date:** {snapshot.get("rebalance_reference_date") or "n/a"}
                - **Coverage:** {_percent(snapshot.get("rebalance_coverage_ratio"))}
                - **Method:** `{snapshot.get("rebalance_method") or "n/a"}`
                - **ACWI conversion scale:** {_number(snapshot.get("rebalance_acwi_conversion_scale"), 8)}
                - **ACWI calibration names:** {snapshot.get("rebalance_acwi_calibration_count") or 0}
                - **Additions:** {", ".join(additions) or "none"}
                - **Removals:** {", ".join(removals) or "none"}
                """
            )
            st.caption(
                "Weight constraints are deterministic. Composition is a "
                "public-data simulation, not an official Nasdaq review: some "
                "prior-rank flags, eligibility inputs and Nasdaq discretion "
                "cannot be independently replicated."
            )
            st.markdown(
                "[Official Nasdaq-100 methodology]"
                "(https://indexes.nasdaqomx.com/docs/Methodology_NDX.pdf)"
            )
    return bool(show_rebalanced)


def _render_source_status(snapshot: dict[str, object]) -> None:
    is_complete = snapshot["status"] == "complete"
    status_label = "Complete live snapshot" if is_complete else "Partial live snapshot"
    dot_class = "" if is_complete else " partial"
    source_line = (
        f"{status_label} | {snapshot.get('reference_fund') or 'No reference fund'} "
        f"| holdings {_display_date(snapshot.get('holdings_as_of'))}"
    )
    status_columns = st.columns([4.7, 1.1], gap="small", vertical_alignment="center")
    with status_columns[0]:
        st.html(
            f"""
            <div class="ndx-source-row" title="{html.escape(source_line)}">
              <span class="ndx-status-dot{dot_class}"></span>{html.escape(source_line)}
            </div>
            """
        )
    with status_columns[1]:
        with st.popover(
            "Data details",
            icon=":material/database:",
            width="stretch",
        ):
            st.markdown(
                f"""
                - **Status:** `{snapshot["status"]}`
                - **Reference fund:** {snapshot.get("reference_fund") or "n/a"}
                - **Holdings source:** `{snapshot.get("holdings_source") or "n/a"}`
                - **Holdings date:** {_display_date(snapshot.get("holdings_as_of"))}
                - **ACWI date:** {_display_date(snapshot.get("reference_data_as_of"))}
                - **Snapshot UTC:** `{snapshot["timestamp"]}`
                """
            )
            if snapshot.get("source_failures"):
                st.caption("Rejected preceding sources")
                st.code(str(snapshot["source_failures"]))

    invalid_count = int(snapshot.get("invalid_float_count", 0))
    if invalid_count:
        st.html(
            f"""
            <span class="ndx-inline-note">
              {invalid_count} inconsistent yfinance fallback excluded
            </span>
            """
        )


def _valid_components(components: pd.DataFrame) -> pd.DataFrame:
    return components.loc[
        components["data_status"].astype("string").str.startswith("valid", na=False)
    ].copy()


def _components_for_view(
    components: pd.DataFrame,
    *,
    rebalanced_view: bool,
) -> pd.DataFrame:
    if not rebalanced_view:
        return components.copy()
    data = components.copy()
    for column in [
        "rebalance_weight",
        "rebalance_reference_weight",
        "rebalance_weight_change",
        "rebalance_weight_delta",
        "rebalance_distortion_contribution",
    ]:
        if column not in data:
            data[column] = pd.NA
        data[column] = pd.to_numeric(data[column], errors="coerce")
    membership = data.get(
        "rebalance_membership",
        pd.Series(False, index=data.index),
    ).fillna(False).astype(bool)
    valid = (
        membership
        & data["rebalance_weight"].notna()
        & data["rebalance_reference_weight"].notna()
    )
    data["actual_weight"] = data["rebalance_weight"]
    data["counterfactual_weight"] = data["rebalance_reference_weight"]
    data["weight_delta"] = data["rebalance_weight_delta"]
    data["distortion_contribution"] = data[
        "rebalance_distortion_contribution"
    ]
    data["weight_ratio"] = (
        data["rebalance_weight"] / data["rebalance_reference_weight"]
    )
    data["data_status"] = "removed_by_rebalance"
    data.loc[membership & ~valid, "data_status"] = "rebalance_missing_inputs"
    data.loc[valid, "data_status"] = "valid_rebalance"
    return data


def _outside_label_axis_range(
    values: pd.Series,
    *,
    padding_ratio: float = 0.24,
) -> list[float]:
    """Reserve horizontal plot space for labels placed past bar endpoints."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return [-0.01, 0.01]
    lower = min(0.0, float(numeric.min()))
    upper = max(0.0, float(numeric.max()))
    span = max(upper - lower, 0.01)
    padding = padding_ratio * span
    return [lower - padding, upper + padding]


def _active_share_view_frame(
    components: pd.DataFrame,
    *,
    rebalanced_view: bool,
) -> tuple[pd.DataFrame, str, str]:
    frame = components.copy()
    ndx_column = (
        "rebalanced_ndx_weight" if rebalanced_view else "ndx_weight"
    )
    delta_column = (
        "rebalanced_weight_delta" if rebalanced_view else "weight_delta"
    )
    for column in [ndx_column, "spx_weight", delta_column]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[ndx_column, "spx_weight", delta_column])
    return frame, ndx_column, delta_column


def _render_active_share_strip(
    summary: dict[str, object],
    *,
    rebalanced_view: bool,
) -> None:
    current_score = float(summary["active_share"])
    rebalanced_score = summary.get("rebalanced_active_share")
    selected_score = (
        float(rebalanced_score)
        if rebalanced_view
        and rebalanced_score is not None
        and not pd.isna(rebalanced_score)
        else current_score
    )
    scenario_label = (
        "Annual-reconstitution NDX weights"
        if rebalanced_view
        else "Published NDX weights"
    )
    annual_value = (
        f"{float(rebalanced_score):.2%}"
        if rebalanced_score is not None and not pd.isna(rebalanced_score)
        else "n/a"
    )
    pair = (
        f"{summary.get('reference_fund') or 'NDX ETF'} vs "
        f"{summary.get('spx_reference_fund') or 'SPX ETF'}"
    )
    st.markdown(
        f"""
        <div class="ndx-active-strip">
          <div class="ndx-active-main">
            <div class="ndx-eyebrow">NDX vs S&amp;P 500 Active Share</div>
            <div class="ndx-active-value">{selected_score:.2%}</div>
            <div class="ndx-score-context">{html.escape(scenario_label)}</div>
          </div>
          <div class="ndx-active-meaning">
            <strong>{selected_score:.2%} of portfolio weight</strong> would need
            to be reallocated to transform one index exposure into the other.
          </div>
          <div class="ndx-stat">
            <div class="ndx-stat-label">ETF pair</div>
            <div class="ndx-stat-value">{html.escape(pair)}</div>
          </div>
          <div class="ndx-stat">
            <div class="ndx-stat-label">If annual reconstitution ran today</div>
            <div class="ndx-stat-value">{annual_value}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _active_share_extreme_figure(
    frame: pd.DataFrame,
    *,
    delta_column: str,
    ndx_column: str,
    side: str,
) -> go.Figure:
    if side == "ndx":
        chart = frame.loc[frame[delta_column].gt(0)].nlargest(
            10,
            delta_column,
        )
        chart = chart.assign(display_delta=chart[delta_column])
        color = NDX_ACTIVE_COLOR
        difference_label = "NDX overweight"
    else:
        chart = frame.loc[frame[delta_column].lt(0)].nsmallest(
            10,
            delta_column,
        )
        chart = chart.assign(display_delta=-chart[delta_column])
        color = SPX_ACTIVE_COLOR
        difference_label = "S&P 500 overweight"
    chart = chart.sort_values("display_delta")
    figure = go.Figure(
        go.Bar(
            x=chart["display_delta"],
            y=chart["ticker"],
            orientation="h",
            marker={"color": color, "line": {"width": 0}},
            text=chart["display_delta"].map(lambda value: f"+{value:.2%}"),
            textposition="outside",
            textfont={"color": color, "size": 10},
            cliponaxis=False,
            customdata=chart[[ndx_column, "spx_weight"]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"{difference_label}: %{{x:.2%}}<br>"
                "NDX weight: %{customdata[0]:.2%}<br>"
                "S&P 500 weight: %{customdata[1]:.2%}"
                "<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        template="plotly_dark" if IS_DARK_MODE else "plotly_white",
        height=340,
        margin={"l": 8, "r": 22, "t": 8, "b": 30},
        showlegend=False,
        bargap=0.3,
        barcornerradius=6,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": THEME["chart_font"], "size": 11},
        xaxis={
            "title": None,
            "tickformat": ".1%",
            "gridcolor": THEME["chart_grid"],
            "zeroline": False,
            "range": _outside_label_axis_range(
                chart["display_delta"],
                padding_ratio=0.30,
            ),
        },
        yaxis={"title": None, "showgrid": False},
    )
    return figure


def _render_active_share_extremes(
    components: pd.DataFrame,
    *,
    rebalanced_view: bool,
) -> None:
    frame, ndx_column, delta_column = _active_share_view_frame(
        components,
        rebalanced_view=rebalanced_view,
    )
    st.subheader("Largest index weight differences")
    st.caption(
        "The strongest single-security tilts in each direction across the "
        "complete NDX and S&P 500 holdings union."
    )
    columns = st.columns(2, gap="large")
    with columns[0]:
        st.markdown("**NDX overweights**")
        st.plotly_chart(
            _active_share_extreme_figure(
                frame,
                delta_column=delta_column,
                ndx_column=ndx_column,
                side="ndx",
            ),
            width="stretch",
            config={"displayModeBar": False},
        )
    with columns[1]:
        st.markdown("**S&P 500 overweights**")
        st.plotly_chart(
            _active_share_extreme_figure(
                frame,
                delta_column=delta_column,
                ndx_column=ndx_column,
                side="spx",
            ),
            width="stretch",
            config={"displayModeBar": False},
        )


def _render_active_share_top_x(
    components: pd.DataFrame,
    *,
    rebalanced_view: bool,
) -> None:
    frame, ndx_column, _ = _active_share_view_frame(
        components,
        rebalanced_view=rebalanced_view,
    )
    ranked = frame.loc[frame[ndx_column].gt(0)].sort_values(
        ndx_column,
        ascending=False,
    )
    if ranked.empty:
        return
    st.subheader("NDX top-constituent concentration")
    st.caption(
        "Select the largest NDX constituents, compare their aggregate weight "
        "in both indices, then inspect the security-level overlap."
    )
    maximum = min(100, len(ranked))
    default = min(20, maximum)
    selected_count = st.slider(
        "Number of top NDX constituents",
        min_value=1,
        max_value=maximum,
        value=default,
        step=1,
        key=f"active_share_top_x_{'annual' if rebalanced_view else 'live'}",
    )
    selected = ranked.head(selected_count).copy()
    ndx_total = float(selected[ndx_column].sum())
    spx_total = float(selected["spx_weight"].sum())
    delta_total = ndx_total - spx_total
    metrics = st.columns([1, 1, 1.2], gap="small")
    metrics[0].metric(
        f"Top {selected_count} in NDX",
        f"{ndx_total:.2%}",
    )
    metrics[1].metric(
        f"Same names in S&P 500",
        f"{spx_total:.2%}",
    )
    metrics[2].metric(
        "Concentration difference",
        f"{delta_total:+.2%}",
    )

    chart = selected.sort_values(ndx_column).copy()
    chart["weight_advantage"] = (
        chart[ndx_column] - chart["spx_weight"]
    ).abs()
    chart["shared_weight"] = chart[[ndx_column, "spx_weight"]].min(axis=1)
    chart["label_anchor"] = chart[[ndx_column, "spx_weight"]].max(axis=1)
    maximum_weight = float(chart["label_anchor"].max())
    label_offset = max(maximum_weight * 0.018, 0.00015)
    chart["label_x"] = chart["label_anchor"] + label_offset
    chart["dominant_index"] = np.where(
        chart[ndx_column].ge(chart["spx_weight"]),
        "NDX",
        "S&P 500",
    )
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=chart["spx_weight"],
            y=chart["ticker"],
            orientation="h",
            name="S&P 500",
            marker={"color": SPX_ACTIVE_COLOR, "line": {"width": 0}},
            opacity=0.9,
            width=0.74,
            customdata=chart[[ndx_column]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "S&P 500 weight: %{x:.2%}<br>"
                "NDX weight: %{customdata[0]:.2%}"
                "<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Bar(
            x=chart[ndx_column],
            y=chart["ticker"],
            orientation="h",
            name=(
                "Annual NDX"
                if rebalanced_view
                else "Published NDX"
            ),
            marker={"color": NDX_ACTIVE_COLOR, "line": {"width": 0}},
            opacity=0.8,
            width=0.42,
            customdata=chart[["spx_weight"]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "NDX weight: %{x:.2%}<br>"
                "S&P 500 weight: %{customdata[0]:.2%}"
                "<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Bar(
            x=chart["shared_weight"],
            y=chart["ticker"],
            orientation="h",
            name="Shared weight",
            marker={"color": ACTIVE_OVERLAP_COLOR, "line": {"width": 0}},
            opacity=0.94,
            width=0.42,
            customdata=chart[[ndx_column, "spx_weight"]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Shared weight: %{x:.2%}<br>"
                "NDX weight: %{customdata[0]:.2%}<br>"
                "S&P 500 weight: %{customdata[1]:.2%}"
                "<extra></extra>"
            ),
        )
    )
    for dominant_index, color in [
        ("NDX", NDX_ACTIVE_COLOR),
        ("S&P 500", SPX_ACTIVE_COLOR),
    ]:
        labels = chart.loc[
            chart["dominant_index"].eq(dominant_index)
            & chart["weight_advantage"].gt(0.00001)
        ]
        figure.add_trace(
            go.Scatter(
                x=labels["label_x"],
                y=labels["ticker"],
                mode="text",
                text=labels["weight_advantage"].map(
                    lambda value: f"+{value:.2%}"
                ),
                textposition="middle right",
                textfont={"color": color, "size": 10},
                cliponaxis=False,
                hoverinfo="skip",
                showlegend=False,
            )
        )
    figure.update_layout(
        template="plotly_dark" if IS_DARK_MODE else "plotly_white",
        height=max(380, 27 * selected_count + 100),
        margin={"l": 8, "r": 18, "t": 18, "b": 35},
        barmode="overlay",
        bargap=0.2,
        barcornerradius=5,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
            "font": {"color": THEME["chart_font"], "size": 11},
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": THEME["chart_font"], "size": 11},
        xaxis={
            "title": None,
            "tickformat": ".1%",
            "gridcolor": THEME["chart_grid"],
            "zeroline": False,
            "range": [0, max(maximum_weight * 1.3, 0.01)],
        },
        yaxis={"title": None, "showgrid": False},
    )
    st.plotly_chart(
        figure,
        width="stretch",
        config={"displayModeBar": False},
    )


def _render_active_share_panel(
    database: SnapshotDatabase,
    snapshot: dict[str, object],
) -> None:
    summary = database.get_active_share(int(snapshot["snapshot_id"]))
    if summary is None:
        st.info(
            "No NDX vs S&P 500 comparison is stored for this snapshot. "
            "Use Refresh to retrieve the matching iShares holdings."
        )
        return
    components = pd.DataFrame(
        database.get_active_share_components(int(snapshot["snapshot_id"]))
    )
    annual_available = (
        summary.get("rebalanced_active_share") is not None
        and not pd.isna(summary.get("rebalanced_active_share"))
    )
    rebalanced_view = st.toggle(
        "Use annual-reconstitution NDX weights",
        value=False,
        disabled=not annual_available,
        help=(
            "Replace the published NDX ETF weights with the simulated full "
            "annual reconstitution. S&P 500 ETF weights remain published."
        ),
    )
    _render_active_share_strip(
        summary,
        rebalanced_view=rebalanced_view,
    )
    st.markdown(
        (
            '<div class="ndx-source-row"><span class="ndx-status-dot"></span>'
            f"{html.escape(str(summary.get('reference_fund') or 'NDX ETF'))} "
            f"holdings {html.escape(str(summary.get('ndx_holdings_as_of') or 'n/a'))}"
            " &nbsp;|&nbsp; "
            f"{html.escape(str(summary.get('spx_reference_fund') or 'SPX ETF'))} "
            f"holdings {html.escape(str(summary.get('spx_holdings_as_of') or 'n/a'))}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    _render_active_share_extremes(
        components,
        rebalanced_view=rebalanced_view,
    )
    _render_active_share_top_x(
        components,
        rebalanced_view=rebalanced_view,
    )


def _render_weight_difference_chart(
    components: pd.DataFrame,
    *,
    rebalanced_view: bool = False,
) -> None:
    valid = _valid_components(components)
    st.subheader("Largest weight differences")
    st.caption(
        (
            "Annual-reconstitution weight minus the selected capitalization "
            "reference."
            if rebalanced_view
            else "Published ETF weight minus the selected capitalization reference."
        )
    )
    if valid.empty:
        st.info("No valid constituent differences are available.")
        return

    chart_frame = (
        valid.nlargest(15, "distortion_contribution")
        .sort_values("weight_delta")
        .copy()
    )
    colors = chart_frame["weight_delta"].map(
        lambda value: "#d45a57" if value >= 0 else "#3f7dc0"
    )
    custom_columns = [
        "actual_weight",
        "counterfactual_weight",
        "distortion_contribution",
    ]
    if rebalanced_view:
        custom_columns.append("rebalance_weight_change")
    custom_data = chart_frame[custom_columns].to_numpy()
    change_labels = (
        chart_frame["rebalance_weight_change"].map(lambda value: f"{value:+.2%}")
        if rebalanced_view
        else None
    )
    change_colors = (
        chart_frame["rebalance_weight_change"].map(
            lambda value: "#268463" if value >= 0 else "#d45a57"
        )
        if rebalanced_view
        else THEME["chart_font"]
    )
    change_hover = (
        "<br>Change vs current: %{customdata[3]:+.2%}"
        if rebalanced_view
        else ""
    )
    displayed_weight_label = (
        "Annual-reconstitution weight" if rebalanced_view else "Published weight"
    )
    figure = go.Figure(
        go.Bar(
            x=chart_frame["weight_delta"],
            y=chart_frame["ticker"],
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            text=change_labels,
            textposition="outside" if rebalanced_view else "none",
            textfont={"color": change_colors, "size": 10},
            cliponaxis=False,
            customdata=custom_data,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Weight difference: %{x:.2%}<br>"
                f"{displayed_weight_label}: %{{customdata[0]:.2%}}<br>"
                "Reference weight: %{customdata[1]:.2%}<br>"
                "WDI contribution: %{customdata[2]:.2f}"
                + change_hover
                + "<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=0, line_width=1, line_color=THEME["chart_zero"])
    figure.update_layout(
        template="plotly_dark" if IS_DARK_MODE else "plotly_white",
        height=430,
        margin={"l": 8, "r": 12, "t": 10, "b": 35},
        showlegend=False,
        bargap=0.28,
        barcornerradius=6,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": THEME["chart_font"], "size": 11},
        xaxis={
            "title": None,
            "tickformat": ".1%",
            "gridcolor": THEME["chart_grid"],
            "zeroline": False,
            "range": (
                _outside_label_axis_range(chart_frame["weight_delta"])
                if rebalanced_view
                else None
            ),
        },
        yaxis={"title": None, "showgrid": False},
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def _render_annual_reconstitution_summary(
    snapshot: dict[str, object],
    analysis: AnnualRebalanceAnalysis,
) -> None:
    score = float(snapshot["rebalance_ndx_wdi"])
    membership = (
        f"{len(analysis.companies)} companies / "
        f"{len(analysis.securities)} securities"
    )
    movement = (
        f"{len(analysis.additions)} in / {len(analysis.removals)} out"
    )
    st.markdown(
        f"""
        <div class="ndx-active-strip">
          <div class="ndx-active-main">
            <div class="ndx-eyebrow">Annual reconstitution today</div>
            <div class="ndx-active-value">{score:.2f}</div>
            <div class="ndx-score-context">Simulated post-review NDX_WDI</div>
          </div>
          <div class="ndx-active-meaning">
            <strong>{analysis.total_capping_redistribution:.2%} of weight</strong>
            is moved by the annual capping rules relative to raw Modified
            Market Cap weights.
          </div>
          <div class="ndx-stat">
            <div class="ndx-stat-label">Current-to-review turnover</div>
            <div class="ndx-stat-value">
              {analysis.current_to_final_turnover:.2%}
            </div>
            <div class="ndx-stat-subvalue">{html.escape(movement)}</div>
          </div>
          <div class="ndx-stat">
            <div class="ndx-stat-label">Selected universe</div>
            <div class="ndx-stat-value">{html.escape(membership)}</div>
            <div class="ndx-stat-subvalue">
              reference {html.escape(str(snapshot.get("rebalance_reference_date") or "n/a"))}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_annual_thresholds(
    analysis: AnnualRebalanceAnalysis,
) -> None:
    thresholds = analysis.thresholds.copy()
    thresholds["distance_label"] = thresholds["distance_to_trigger"].map(
        lambda value: (
            f"{abs(value) * 100:.2f} pp below"
            if value >= 0
            else f"{abs(value) * 100:.2f} pp above"
        )
    )
    thresholds["status_label"] = np.where(
        thresholds["triggered"],
        "Triggered",
        "Not triggered",
    )
    colors = thresholds["triggered"].map(
        {True: NDX_ACTIVE_COLOR, False: "#268463"}
    )
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=thresholds["actual"],
            y=thresholds["label"],
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            text=thresholds.apply(
                lambda row: (
                    f"{row['actual']:.2%} | {row['distance_label']}"
                ),
                axis=1,
            ),
            textposition="outside",
            textfont={"color": colors, "size": 10},
            cliponaxis=False,
            customdata=thresholds[
                ["threshold", "target", "status_label", "rule"]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Observed input: %{x:.2%}<br>"
                "Trigger: %{customdata[0]:.2%}<br>"
                "Adjustment target: %{customdata[1]:.2%}<br>"
                "Status: %{customdata[2]}<br>"
                "%{customdata[3]}"
                "<extra></extra>"
            ),
            name="Observed input",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=thresholds["threshold"],
            y=thresholds["label"],
            mode="markers",
            marker={
                "color": "#d69a2d",
                "size": 11,
                "symbol": "diamond",
                "line": {"color": THEME["marker_outline"], "width": 1},
            },
            customdata=thresholds[["rule"]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Trigger threshold: %{x:.2%}<br>"
                "%{customdata[0]}<extra></extra>"
            ),
            name="Rule trigger",
        )
    )
    maximum = float(
        thresholds[["actual", "threshold"]].to_numpy().max()
    )
    figure.update_layout(
        template="plotly_dark" if IS_DARK_MODE else "plotly_white",
        height=350,
        margin={"l": 8, "r": 165, "t": 8, "b": 35},
        barmode="overlay",
        bargap=0.38,
        barcornerradius=6,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"color": THEME["chart_font"], "size": 10},
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": THEME["chart_font"], "size": 11},
        xaxis={
            "title": None,
            "tickformat": ".0%",
            "range": [0, max(0.65, maximum * 1.28)],
            "gridcolor": THEME["chart_grid"],
            "zeroline": False,
        },
        yaxis={"title": None, "showgrid": False},
    )
    st.subheader("Distance to annual capping triggers")
    st.caption(
        "Bars are the exact inputs tested at each stage. Diamonds mark the "
        "official trigger; negative distance means the rule is active."
    )
    st.plotly_chart(
        figure,
        width="stretch",
        config={"displayModeBar": False},
    )


def _annual_cumulative_weights_figure(
    analysis: AnnualRebalanceAnalysis,
) -> go.Figure:
    companies = analysis.companies.sort_values(
        "initial_weight",
        ascending=False,
    ).copy()
    companies["rank"] = np.arange(1, len(companies) + 1)
    companies["initial_cumulative"] = companies["initial_weight"].cumsum()
    companies["final_cumulative"] = companies["final_weight"].cumsum()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=companies["rank"],
            y=companies["initial_cumulative"],
            mode="lines",
            name="Initial Modified Market Cap",
            line={"color": NDX_ACTIVE_COLOR, "width": 3},
            customdata=companies[
                ["company_name", "tickers", "initial_weight"]
            ].to_numpy(),
            hovertemplate=(
                "Rank %{x}<br>"
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "Initial company weight: %{customdata[2]:.2%}<br>"
                "Cumulative weight: %{y:.2%}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=companies["rank"],
            y=companies["final_cumulative"],
            mode="lines",
            name="After annual capping",
            line={"color": "#268463", "width": 2},
            customdata=companies[
                ["company_name", "tickers", "final_weight"]
            ].to_numpy(),
            hovertemplate=(
                "Rank %{x}<br>"
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "Final company weight: %{customdata[2]:.2%}<br>"
                "Cumulative weight: %{y:.2%}<extra></extra>"
            ),
        )
    )
    for rank, label in [(75, "Top 75"), (100, "100 selected")]:
        if rank <= len(companies):
            figure.add_vline(
                x=rank,
                line_width=1,
                line_dash="dot",
                line_color=THEME["chart_zero"],
                annotation_text=label,
                annotation_position="top left",
            )
    figure.update_layout(
        template="plotly_dark" if IS_DARK_MODE else "plotly_white",
        height=430,
        margin={"l": 8, "r": 20, "t": 35, "b": 40},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.03,
            "xanchor": "left",
            "x": 0,
            "font": {"color": THEME["chart_font"], "size": 10},
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": THEME["chart_font"], "size": 11},
        xaxis={
            "title": "Company rank by Modified Market Cap",
            "showgrid": False,
            "range": [1, max(100, len(companies))],
        },
        yaxis={
            "title": "Cumulative weight",
            "tickformat": ".0%",
            "range": [0, 1.02],
            "gridcolor": THEME["chart_grid"],
            "zeroline": False,
        },
    )
    return figure


def _annual_float_multiple_figure(
    analysis: AnnualRebalanceAnalysis,
) -> go.Figure:
    frame = analysis.securities.sort_values(
        "initial_weight",
        ascending=False,
    ).copy()
    frame["rank"] = np.arange(1, len(frame) + 1)
    frame["modified_cap_ratio"] = pd.to_numeric(
        frame["modified_cap_ratio"],
        errors="coerce",
    )
    sizes = 7 + 28 * (
        frame["initial_weight"] / frame["initial_weight"].max()
    )
    colors = np.where(
        frame["modified_cap_ratio"].ge(3.0 - 1e-8),
        NDX_ACTIVE_COLOR,
        SPX_ACTIVE_COLOR,
    )
    figure = go.Figure(
        go.Scatter(
            x=frame["rank"],
            y=frame["modified_cap_ratio"],
            mode="markers",
            marker={
                "color": colors,
                "size": sizes,
                "opacity": 0.78,
                "line": {"color": THEME["marker_outline"], "width": 0.8},
            },
            customdata=frame[
                [
                    "ticker",
                    "company_name",
                    "initial_weight",
                    "rebalance_input_status",
                ]
            ].to_numpy(),
            hovertemplate=(
                "Rank %{x}<br>"
                "<b>%{customdata[0]}</b> - %{customdata[1]}<br>"
                "TSO / free-float multiple used: %{y:.2f}x<br>"
                "Initial Modified Market Cap weight: "
                "%{customdata[2]:.2%}<br>"
                "Input: %{customdata[3]}<extra></extra>"
            ),
        )
    )
    figure.add_hline(
        y=3.0,
        line_width=1,
        line_dash="dash",
        line_color=NDX_ACTIVE_COLOR,
        annotation_text="3x free-float ceiling",
        annotation_position="top right",
    )
    figure.update_layout(
        template="plotly_dark" if IS_DARK_MODE else "plotly_white",
        height=430,
        margin={"l": 8, "r": 20, "t": 35, "b": 40},
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": THEME["chart_font"], "size": 11},
        xaxis={
            "title": "Security rank by initial weight",
            "showgrid": False,
        },
        yaxis={
            "title": "Modified-cap multiple",
            "ticksuffix": "x",
            "range": [0.85, 3.2],
            "gridcolor": THEME["chart_grid"],
            "zeroline": False,
        },
    )
    return figure


def _annual_cumulative_transfer_figure(
    analysis: AnnualRebalanceAnalysis,
) -> go.Figure:
    frame = analysis.securities.sort_values(
        "initial_weight",
        ascending=False,
    ).copy()
    frame["rank"] = np.arange(1, len(frame) + 1)
    frame["cumulative_transfer"] = frame["capping_change"].cumsum()
    figure = go.Figure(
        go.Scatter(
            x=frame["rank"],
            y=frame["cumulative_transfer"],
            mode="lines",
            line={"color": NDX_ACTIVE_COLOR, "width": 3},
            fill="tozeroy",
            fillcolor="rgba(223,107,79,0.16)",
            customdata=frame[
                ["ticker", "initial_weight", "final_weight", "capping_change"]
            ].to_numpy(),
            hovertemplate=(
                "Initial rank %{x}<br>"
                "<b>%{customdata[0]}</b><br>"
                "Initial weight: %{customdata[1]:.2%}<br>"
                "Final weight: %{customdata[2]:.2%}<br>"
                "Security change: %{customdata[3]:+.2%}<br>"
                "Cumulative transfer: %{y:+.2%}<extra></extra>"
            ),
        )
    )
    minimum_row = frame.loc[frame["cumulative_transfer"].idxmin()]
    figure.add_annotation(
        x=float(minimum_row["rank"]),
        y=float(minimum_row["cumulative_transfer"]),
        text=(
            f"{abs(float(minimum_row['cumulative_transfer'])):.2%} shifted "
            "out of higher ranks"
        ),
        showarrow=True,
        arrowhead=2,
        ax=45,
        ay=45,
        font={"color": NDX_ACTIVE_COLOR, "size": 10},
    )
    figure.add_hline(
        y=0,
        line_width=1,
        line_color=THEME["chart_zero"],
    )
    figure.update_layout(
        template="plotly_dark" if IS_DARK_MODE else "plotly_white",
        height=420,
        margin={"l": 8, "r": 20, "t": 25, "b": 40},
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": THEME["chart_font"], "size": 11},
        xaxis={
            "title": "Initial Modified Market Cap rank",
            "showgrid": False,
        },
        yaxis={
            "title": "Cumulative final minus initial weight",
            "tickformat": ".1%",
            "gridcolor": THEME["chart_grid"],
            "zeroline": False,
        },
    )
    return figure


def _annual_largest_transfers_figure(
    analysis: AnnualRebalanceAnalysis,
) -> go.Figure:
    frame = analysis.securities.copy()
    donors = frame.nsmallest(10, "capping_change")
    beneficiaries = frame.nlargest(10, "capping_change")
    display = (
        pd.concat([donors, beneficiaries], ignore_index=True)
        .drop_duplicates("ticker")
        .sort_values("capping_change")
    )
    colors = display["capping_change"].map(
        lambda value: "#268463" if value >= 0 else NDX_ACTIVE_COLOR
    )
    figure = go.Figure(
        go.Bar(
            x=display["capping_change"],
            y=display["ticker"],
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            text=display["capping_change"].map(
                lambda value: f"{value:+.2%}"
            ),
            textposition="outside",
            textfont={"color": colors, "size": 10},
            cliponaxis=False,
            customdata=display[
                ["initial_weight", "company_stage_weight", "final_weight"]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Capping change: %{x:+.2%}<br>"
                "Initial weight: %{customdata[0]:.2%}<br>"
                "After company stage: %{customdata[1]:.2%}<br>"
                "Final weight: %{customdata[2]:.2%}"
                "<extra></extra>"
            ),
        )
    )
    figure.add_vline(
        x=0,
        line_width=1,
        line_color=THEME["chart_zero"],
    )
    figure.update_layout(
        template="plotly_dark" if IS_DARK_MODE else "plotly_white",
        height=520,
        margin={"l": 8, "r": 42, "t": 15, "b": 35},
        showlegend=False,
        bargap=0.25,
        barcornerradius=5,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": THEME["chart_font"], "size": 11},
        xaxis={
            "title": None,
            "tickformat": ".1%",
            "gridcolor": THEME["chart_grid"],
            "zeroline": False,
            "range": _outside_label_axis_range(
                display["capping_change"],
                padding_ratio=0.32,
            ),
        },
        yaxis={"title": None, "showgrid": False},
    )
    return figure


def _render_annual_reconstitution_panel(
    snapshot: dict[str, object],
    components: pd.DataFrame,
) -> None:
    if snapshot.get("rebalance_ndx_wdi") is None:
        st.info(
            "No annual-reconstitution simulation is stored for this snapshot. "
            "Use Refresh to calculate one."
        )
        return
    try:
        analysis = analyze_annual_rebalance(components)
    except ValueError as exc:
        st.warning(f"Annual-reconstitution analysis is unavailable: {exc}")
        return

    _render_annual_reconstitution_summary(snapshot, analysis)
    st.markdown(
        (
            '<div class="ndx-source-row"><span class="ndx-status-dot"></span>'
            f"{html.escape(str(snapshot.get('rebalance_method') or 'annual simulation'))}"
            " &nbsp;|&nbsp; "
            f"{html.escape(str(snapshot.get('rebalance_data_source') or 'public data'))}"
            " &nbsp;|&nbsp; "
            f"coverage {_percent(snapshot.get('rebalance_coverage_ratio'))}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    if (
        analysis.persisted_weight_error is not None
        and analysis.persisted_weight_error > 1e-8
    ):
        st.warning(
            "The reconstructed stages differ from the persisted final weights "
            f"by {analysis.persisted_weight_error:.3g}."
        )

    _render_annual_thresholds(analysis)

    st.subheader("Modified Market Cap inputs and concentration")
    st.caption(
        "The selected company distribution is shown before and after capping. "
        "Candidate companies outside the selected universe are not persisted, "
        "so no synthetic rank 101-125 distance is inferred."
    )
    input_columns = st.columns(2, gap="large")
    with input_columns[0]:
        st.markdown("**Cumulative selected-company weights**")
        st.plotly_chart(
            _annual_cumulative_weights_figure(analysis),
            width="stretch",
            config={"displayModeBar": False},
        )
    with input_columns[1]:
        st.markdown("**Float adjustment used in Modified Market Cap**")
        st.plotly_chart(
            _annual_float_multiple_figure(analysis),
            width="stretch",
            config={"displayModeBar": False},
        )

    st.subheader("Weight transferred by the annual rules")
    metrics = st.columns(5, gap="small")
    metrics[0].metric(
        "Company capping",
        f"{analysis.company_redistribution:.2%}",
    )
    metrics[1].metric(
        "Security capping",
        f"{analysis.security_redistribution:.2%}",
    )
    metrics[2].metric(
        "Total capping transfer",
        f"{analysis.total_capping_redistribution:.2%}",
    )
    metrics[3].metric(
        "Rank order preserved",
        (
            f"{min(analysis.company_rank_preservation_ratio, analysis.security_rank_preservation_ratio):.2%}"
        ),
        delta=(
            f"company {analysis.company_rank_preservation_ratio:.2%} | "
            f"security {analysis.security_rank_preservation_ratio:.2%}"
        ),
        delta_color="off",
    )
    metrics[4].metric(
        "Recipients / donors",
        f"{analysis.beneficiary_count} / {analysis.donor_count}",
    )
    transfer_columns = st.columns([1.08, 0.92], gap="large")
    with transfer_columns[0]:
        st.markdown("**Cumulative transfer by initial rank**")
        st.plotly_chart(
            _annual_cumulative_transfer_figure(analysis),
            width="stretch",
            config={"displayModeBar": False},
        )
    with transfer_columns[1]:
        st.markdown("**Largest capping donors and recipients**")
        st.plotly_chart(
            _annual_largest_transfers_figure(analysis),
            width="stretch",
            config={"displayModeBar": False},
        )

    change_columns = st.columns([1.12, 0.88], gap="large")
    with change_columns[0]:
        _render_rebalance_changes_chart(components)
    with change_columns[1]:
        _render_rebalance_membership_chart(components)

    with st.expander(
        f"Annual weighting audit ({len(analysis.securities)} securities)"
    ):
        audit = analysis.securities[
            [
                "ticker",
                "company_name",
                "initial_rank",
                "initial_weight",
                "company_stage_weight",
                "final_weight",
                "capping_change",
                "actual_weight",
                "current_change",
                "modified_cap_ratio",
                "rebalance_input_status",
            ]
        ].sort_values("initial_rank")
        st.dataframe(
            audit,
            hide_index=True,
            width="stretch",
            height=560,
        )


def _render_rebalance_changes_chart(components: pd.DataFrame) -> None:
    if "rebalance_weight_change" not in components:
        return
    frame = components.copy()
    frame["rebalance_weight_change"] = pd.to_numeric(
        frame["rebalance_weight_change"], errors="coerce"
    )
    frame = (
        frame.dropna(subset=["rebalance_weight_change"])
        .assign(
            absolute_change=lambda data: data["rebalance_weight_change"].abs()
        )
        .nlargest(15, "absolute_change")
        .sort_values("rebalance_weight_change")
    )
    if frame.empty:
        return
    st.subheader("Largest changes caused by the annual reconstitution")
    st.caption(
        "Simulated annual-reconstitution weight minus the current published weight."
    )
    colors = frame["rebalance_weight_change"].map(
        lambda value: "#268463" if value >= 0 else "#d45a57"
    )
    figure = go.Figure(
        go.Bar(
            x=frame["rebalance_weight_change"],
            y=frame["ticker"],
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            text=frame["rebalance_weight_change"].map(
                lambda value: f"{value:+.2%}"
            ),
            textposition="outside",
            textfont={"color": colors, "size": 10},
            cliponaxis=False,
            customdata=frame[["actual_weight", "rebalance_weight"]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Change: %{x:+.2%}<br>"
                "Current weight: %{customdata[0]:.2%}<br>"
                "Annual-reconstitution weight: %{customdata[1]:.2%}"
                "<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=0, line_width=1, line_color=THEME["chart_zero"])
    figure.update_layout(
        template="plotly_dark" if IS_DARK_MODE else "plotly_white",
        height=430,
        margin={"l": 8, "r": 35, "t": 10, "b": 35},
        showlegend=False,
        bargap=0.28,
        barcornerradius=6,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": THEME["chart_font"], "size": 11},
        xaxis={
            "title": None,
            "tickformat": ".1%",
            "gridcolor": THEME["chart_grid"],
            "zeroline": False,
            "range": _outside_label_axis_range(
                frame["rebalance_weight_change"]
            ),
        },
        yaxis={"title": None, "showgrid": False},
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def _render_rebalance_membership_chart(components: pd.DataFrame) -> None:
    """Show simulated additions and removals with their entering/exiting weight."""
    if "rebalance_membership" not in components:
        return
    frame = components.copy()
    frame["actual_weight"] = pd.to_numeric(frame["actual_weight"], errors="coerce").fillna(0)
    frame["rebalance_weight"] = pd.to_numeric(
        frame.get("rebalance_weight"), errors="coerce"
    ).fillna(0)
    membership = frame["rebalance_membership"].fillna(False).astype(bool)
    entries = membership & frame["actual_weight"].le(0) & frame["rebalance_weight"].gt(0)
    exits = ~membership & frame["actual_weight"].gt(0)
    frame = frame.loc[entries | exits].copy()

    st.subheader("Index entries and exits")
    st.caption(
        "Entry bars show simulated annual-reconstitution weight; exit bars show "
        "current weight removed from the index."
    )
    if frame.empty:
        st.info("No index additions or removals are produced by today's simulation.")
        return

    frame["movement"] = np.where(
        frame["rebalance_membership"].astype(bool),
        "Entry",
        "Exit",
    )
    frame["membership_weight"] = np.where(
        frame["movement"].eq("Entry"),
        frame["rebalance_weight"],
        -frame["actual_weight"],
    )
    frame = frame.sort_values("membership_weight")
    colors = frame["movement"].map({"Entry": "#268463", "Exit": "#d45a57"})
    figure = go.Figure(
        go.Bar(
            x=frame["membership_weight"],
            y=frame["ticker"],
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            text=frame.apply(
                lambda row: (
                    f"Entry {row['rebalance_weight']:.2%}"
                    if row["movement"] == "Entry"
                    else f"Exit {row['actual_weight']:.2%}"
                ),
                axis=1,
            ),
            textposition="outside",
            textfont={"color": colors, "size": 10},
            cliponaxis=False,
            customdata=frame[
                ["movement", "actual_weight", "rebalance_weight"]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "%{customdata[0]}<br>"
                "Current weight: %{customdata[1]:.2%}<br>"
                "Annual-reconstitution weight: %{customdata[2]:.2%}"
                "<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=0, line_width=1, line_color=THEME["chart_zero"])
    figure.update_layout(
        template="plotly_dark" if IS_DARK_MODE else "plotly_white",
        height=max(250, 54 * len(frame) + 90),
        margin={"l": 8, "r": 70, "t": 10, "b": 35},
        showlegend=False,
        bargap=0.34,
        barcornerradius=6,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": THEME["chart_font"], "size": 11},
        xaxis={
            "title": None,
            "tickformat": ".1%",
            "gridcolor": THEME["chart_grid"],
            "zeroline": False,
            "range": _outside_label_axis_range(
                frame["membership_weight"],
                padding_ratio=0.40,
            ),
        },
        yaxis={"title": None, "showgrid": False},
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def _load_quarterly_history() -> pd.DataFrame:
    history_path = os.getenv(
        "NDX_QUARTERLY_HISTORY_PATH",
        "data/edgar_quarterly_history.csv",
    )
    if not os.path.exists(history_path):
        return pd.DataFrame()
    history = pd.read_csv(history_path)
    required = {
        "report_date",
        "ndx_wdi",
        "ndx_wdi_raw",
        "coverage_ratio",
        "matched_count",
        "estimated_count",
        "excluded_non_comparable_count",
        "rebalance_type",
    }
    if required.difference(history.columns):
        return pd.DataFrame()
    history["report_date"] = pd.to_datetime(history["report_date"], errors="coerce")
    return history.dropna(subset=["report_date", "ndx_wdi"]).sort_values(
        "report_date"
    )


def _render_history_help() -> None:
    with st.popover("History method", icon=":material/help:", width="stretch"):
        st.markdown(
            """
            One observation is retained per public quarter. QQQ and SPGM positions
            are matched by exact CUSIP. Missing comparable SPGM weights are estimated
            from the median observed overweight ratio; ADR and listing-form mismatches
            remain excluded.

            The quarterly series and the live point use different reference sources:
            SEC-filed SPGM for history and the current iShares ACWI portfolio for live
            monitoring.
            """
        )
        st.caption(
            "Diamond markers identify annual December reconstitutions. The star "
            "marks the September 2023 quarter-end observation following the July "
            "2023 special rebalance."
        )


def _render_quarterly_history(snapshot: dict[str, object]) -> None:
    history = _load_quarterly_history()
    header_columns = st.columns([3.5, 1.1], gap="small", vertical_alignment="bottom")
    with header_columns[0]:
        st.subheader("Quarterly history and live reading")
        st.caption("Corrected QQQ/SPGM filings, extended with the current ACWI reading.")
    with header_columns[1]:
        _render_history_help()

    if history.empty:
        st.info(
            "Quarterly SEC history is unavailable. Run "
            "`python run_quarterly_history.py` to rebuild it."
        )
        return

    custom_data = history[
        [
            "ndx_wdi_raw",
            "coverage_ratio",
            "matched_count",
            "estimated_count",
            "excluded_non_comparable_count",
        ]
    ].to_numpy()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=history["report_date"],
            y=history["ndx_wdi"],
            mode="lines+markers",
            name="Quarterly SEC",
            line={"color": "#177e78", "width": 2.4},
            marker={
                "size": 5.5,
                "color": THEME["marker_outline"],
                "line": {"color": "#177e78", "width": 1.5},
            },
            customdata=custom_data,
            hovertemplate=(
                "<b>%{x|%b %Y}</b><br>"
                "Corrected NDX_WDI: %{y:.2f}<br>"
                "Matched-only NDX_WDI: %{customdata[0]:.2f}<br>"
                "QQQ weight observed: %{customdata[1]:.1%}<br>"
                "Matched / estimated: %{customdata[2]:.0f} / %{customdata[3]:.0f}<br>"
                "Non-comparable exclusions: %{customdata[4]:.0f}"
                "<extra></extra>"
            ),
        )
    )

    annual = history.loc[
        history["rebalance_type"].eq("annual_reconstitution")
    ]
    figure.add_trace(
        go.Scatter(
            x=annual["report_date"],
            y=annual["ndx_wdi"],
            mode="markers",
            showlegend=False,
            marker={
                "symbol": "diamond",
                "size": 8,
                "color": "#d69a2d",
                "line": {"color": THEME["marker_outline"], "width": 1},
            },
            hovertemplate=(
                "<b>Annual reconstitution</b><br>"
                "%{x|%b %Y}<br>NDX_WDI: %{y:.2f}<extra></extra>"
            ),
        )
    )

    special = history.loc[history["rebalance_type"].eq("special_rebalance")]
    figure.add_trace(
        go.Scatter(
            x=special["report_date"],
            y=special["ndx_wdi"],
            mode="markers",
            showlegend=False,
            marker={
                "symbol": "star",
                "size": 12,
                "color": "#d45a57",
                "line": {"color": THEME["marker_outline"], "width": 1},
            },
            hovertemplate=(
                "<b>Quarter after the July 2023 special rebalance</b><br>"
                "%{x|%b %Y}<br>NDX_WDI: %{y:.2f}<extra></extra>"
            ),
        )
    )

    live_date = pd.to_datetime(snapshot["timestamp"], errors="coerce")
    if pd.isna(live_date):
        live_date = pd.Timestamp.now()
    if getattr(live_date, "tzinfo", None) is not None:
        live_date = live_date.tz_localize(None)
    live_score = float(snapshot["ndx_wdi"])
    last_history = history.iloc[-1]
    figure.add_trace(
        go.Scatter(
            x=[last_history["report_date"], live_date],
            y=[last_history["ndx_wdi"], live_score],
            mode="lines",
            showlegend=False,
            hoverinfo="skip",
            line={"color": "#268463", "width": 1.6, "dash": "dot"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[live_date],
            y=[live_score],
            mode="markers",
            name="Live ACWI",
            marker={
                "size": 11,
                "color": "#268463",
                "line": {"color": THEME["marker_outline"], "width": 2},
            },
            customdata=[
                [
                    snapshot.get("coverage_ratio"),
                    snapshot.get("reference_fund"),
                    snapshot.get("reference_data_as_of"),
                ]
            ],
            hovertemplate=(
                "<b>Live ACWI reading</b><br>"
                "%{x|%b %d, %Y}<br>"
                "NDX_WDI: %{y:.2f}<br>"
                "Coverage: %{customdata[0]:.1%}<br>"
                "Reference fund: %{customdata[1]}<br>"
                "ACWI date: %{customdata[2]}<extra></extra>"
            ),
        )
    )

    all_scores = pd.concat(
        [history["ndx_wdi"], pd.Series([live_score])],
        ignore_index=True,
    )
    figure.add_vline(
        x="2023-07-24",
        line_width=1,
        line_dash="dot",
        line_color="rgba(212,90,87,0.45)",
    )
    figure.update_layout(
        template="plotly_dark" if IS_DARK_MODE else "plotly_white",
        height=430,
        margin={"l": 10, "r": 15, "t": 10, "b": 35},
        hovermode="closest",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 10},
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": THEME["chart_font"], "size": 11},
        xaxis={
            "title": None,
            "dtick": "M12",
            "tickformat": "%Y",
            "showgrid": False,
        },
        yaxis={
            "title": "NDX_WDI",
            "range": [
                max(0.0, float(all_scores.min()) - 1.0),
                float(all_scores.max()) + 1.0,
            ],
            "gridcolor": THEME["chart_grid"],
            "zeroline": False,
        },
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def _render_rankings(
    components: pd.DataFrame,
    weighting_basis: str,
    *,
    rebalanced_view: bool = False,
) -> None:
    valid = _valid_components(components)
    st.html('<div class="ndx-section-rule"></div>')
    st.subheader("Explore constituents")
    overweights, underweights, contributors = st.tabs(
        ["Overweights", "Underweights", "WDI contributors"]
    )
    with overweights:
        st.dataframe(
            _table_for_display(
                valid.loc[valid["weight_delta"] > 0].nlargest(12, "weight_delta"),
                weighting_basis,
                rebalanced_view=rebalanced_view,
            ),
            hide_index=True,
            width="stretch",
        )
    with underweights:
        st.dataframe(
            _table_for_display(
                valid.loc[valid["weight_delta"] < 0].nsmallest(12, "weight_delta"),
                weighting_basis,
                rebalanced_view=rebalanced_view,
            ),
            hide_index=True,
            width="stretch",
        )
    with contributors:
        st.dataframe(
            _table_for_display(
                valid.nlargest(12, "distortion_contribution"),
                weighting_basis,
                rebalanced_view=rebalanced_view,
            ),
            hide_index=True,
            width="stretch",
        )

    with st.expander(f"All constituents ({len(components)})"):
        st.dataframe(
            _table_for_display(
                components,
                weighting_basis,
                rebalanced_view=rebalanced_view,
            ),
            hide_index=True,
            width="stretch",
        )

    excluded = components.loc[
        ~components["data_status"].astype("string").str.startswith("valid", na=False)
    ]
    if not excluded.empty:
        with st.expander(f"Excluded data ({len(excluded)})"):
            st.dataframe(
                _table_for_display(
                    excluded,
                    weighting_basis,
                    rebalanced_view=rebalanced_view,
                ),
                hide_index=True,
                width="stretch",
            )


title_columns = st.columns([4.6, 0.8], gap="small", vertical_alignment="top")
with title_columns[0]:
    st.title("Nasdaq-100 Analytics")
with title_columns[1]:
    st.button(
        "Light mode" if IS_DARK_MODE else "Night mode",
        icon=":material/light_mode:" if IS_DARK_MODE else ":material/dark_mode:",
        key="theme_toggle",
        width="stretch",
        help="Switch the dashboard color theme.",
        on_click=_toggle_dark_mode,
    )

panel_label = st.segmented_control(
    "Analysis",
    [
        "NDX Distortion Index",
        "NDX vs S&P 500",
        "Annual Reconstitution",
    ],
    default="NDX Distortion Index",
    required=True,
    key="analysis_panel_selector",
    width="stretch",
)
panel_captions = {
    "NDX Distortion Index": (
        "Live ETF weights compared with a pure-capitalization reference."
    ),
    "NDX vs S&P 500": (
        "Security-level Active Share between matching Nasdaq-100 and "
        "S&P 500 iShares ETFs."
    ),
    "Annual Reconstitution": (
        "Visual audit of Modified Market Cap weighting, annual thresholds, "
        "capping transfers, and simulated membership changes."
    ),
}
st.caption(panel_captions[panel_label])

control_widths = (
    [1.05, 1.05, 0.48, 0.62]
    if panel_label == "NDX Distortion Index"
    else [1.2, 0.48, 0.72]
)
control_columns = st.columns(
    control_widths,
    gap="small",
    vertical_alignment="bottom",
)
with control_columns[0]:
    universe_label = st.segmented_control(
        "Universe",
        ["Non-UCITS", "UCITS"],
        default="Non-UCITS",
        required=True,
        key="universe_selector",
        width="stretch",
    )

universe = {"Non-UCITS": "non_ucits", "UCITS": "ucits"}[universe_label]
if panel_label == "NDX Distortion Index":
    with control_columns[1]:
        basis_label = st.segmented_control(
            "Capitalization basis",
            ["Free float", "Total"],
            default="Free float",
            required=True,
            key="basis_selector",
            width="stretch",
        )
    weighting_basis = {"Free float": "float", "Total": "total"}[basis_label]
    refresh_column = control_columns[2]
    help_column = control_columns[3]
else:
    basis_label = "Free float"
    weighting_basis = "float"
    refresh_column = control_columns[1]
    help_column = control_columns[2]

with refresh_column:
    refresh_help = (
        f"Refresh live data for {universe_label}, including the matching "
        "S&P 500 ETF comparison."
        if panel_label == "NDX vs S&P 500"
        else f"Refresh live data and the annual simulation for {universe_label}."
    )
    recompute_clicked = st.button(
        "Refresh",
        icon=":material/refresh:",
        type="primary",
        width="stretch",
        help=refresh_help,
    )
with help_column:
    if panel_label == "NDX Distortion Index":
        _render_method_help(weighting_basis)
    elif panel_label == "NDX vs S&P 500":
        _render_active_share_help()
    else:
        _render_annual_reconstitution_help()

if recompute_clicked:
    with st.spinner(f"Refreshing {universe_label}..."):
        try:
            recompute_snapshot(
                db_path=os.getenv("NDX_DB_PATH"),
                universe=universe,
                weighting_basis=weighting_basis,
            )
            st.session_state["_refresh_notice"] = f"{universe_label} updated"
            st.rerun()
        except Exception as exc:
            st.error(f"Refresh failed: {exc}")

database = _database()
snapshot = database.get_current(universe, weighting_basis)
if snapshot is None:
    st.info(
        f"No {universe_label} snapshot is available for the "
        f"{basis_label.lower()} basis. Use Refresh to retrieve live data."
    )
    st.stop()

if notice := st.session_state.pop("_refresh_notice", None):
    st.toast(notice, icon=":material/check_circle:")

components = pd.DataFrame(database.get_components(int(snapshot["snapshot_id"])))
if panel_label == "Annual Reconstitution":
    _render_annual_reconstitution_panel(snapshot, components)
    st.caption(
        "This is an auditable public-data simulation, not an official Nasdaq "
        "review result. Nasdaq discretion and non-public review flags remain "
        "outside the model."
    )
elif panel_label == "NDX vs S&P 500":
    _render_active_share_panel(database, snapshot)
    st.caption(
        "Active Share uses the complete normalized equity holdings union. "
        "ETF pairs are never merged or averaged across regulatory universes."
    )
else:
    _render_score_strip(
        snapshot,
        components,
        universe_label,
        basis_label,
    )
    show_rebalanced = _render_rebalance_controls(snapshot)
    _render_source_status(snapshot)
    display_components = _components_for_view(
        components,
        rebalanced_view=show_rebalanced,
    )

    show_history = universe == "non_ucits" and weighting_basis == "float"
    if show_history:
        chart_columns = st.columns([0.93, 1.35], gap="large")
        with chart_columns[0]:
            _render_weight_difference_chart(
                display_components,
                rebalanced_view=show_rebalanced,
            )
        with chart_columns[1]:
            _render_quarterly_history(snapshot)
    else:
        _render_weight_difference_chart(
            display_components,
            rebalanced_view=show_rebalanced,
        )

    if show_rebalanced:
        _render_rebalance_changes_chart(components)
        _render_rebalance_membership_chart(components)

    _render_rankings(
        display_components,
        weighting_basis,
        rebalanced_view=show_rebalanced,
    )

    st.caption(
        "Universes are never merged or averaged. Each score retains the reference "
        "fund and holdings source that were actually used."
    )
