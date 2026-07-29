"""Backward-compatible imports for the Active Share domain engine."""

from ndx_wdi.domain.active_share import (
    ActiveShareResult,
    ActiveShareSleeves,
    calculate_active_share,
    calculate_active_share_sleeves,
)

__all__ = [
    "ActiveShareResult",
    "ActiveShareSleeves",
    "calculate_active_share",
    "calculate_active_share_sleeves",
]
