import ast
from pathlib import Path

import active_share
import distortion_engine
import nasdaq100_rebalance
import rebalance_analytics
from ndx_wdi.domain import active_share as domain_active_share
from ndx_wdi.domain import distortion as domain_distortion
from ndx_wdi.domain import rebalance as domain_rebalance
from ndx_wdi.domain import rebalance_analytics as domain_rebalance_analytics


DOMAIN_DIR = Path(__file__).parents[1] / "ndx_wdi" / "domain"
FORBIDDEN_IMPORTS = {
    "background_jobs",
    "database",
    "provider_cache",
    "requests",
    "streamlit",
    "yfinance",
}


def test_domain_modules_do_not_import_io_or_ui_dependencies():
    imported_roots: set[str] = set()
    for path in DOMAIN_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])

    assert imported_roots.isdisjoint(FORBIDDEN_IMPORTS)


def test_legacy_modules_reexport_domain_engines():
    assert active_share.calculate_active_share is domain_active_share.calculate_active_share
    assert distortion_engine.calculate_distortion is domain_distortion.calculate_distortion
    assert (
        nasdaq100_rebalance.simulate_rebalance
        is domain_rebalance.simulate_rebalance
    )
    assert (
        rebalance_analytics.analyze_annual_rebalance
        is domain_rebalance_analytics.analyze_annual_rebalance
    )
