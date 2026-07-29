"""Backward-compatible imports for the distortion domain engine."""

from ndx_wdi.domain.distortion import (
    COMPONENT_COLUMNS,
    DistortionResult,
    calculate_distortion,
    normalize_actual_weights,
)

__all__ = [
    "COMPONENT_COLUMNS",
    "DistortionResult",
    "calculate_distortion",
    "normalize_actual_weights",
]
