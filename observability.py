"""Small, dependency-free performance instrumentation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable


Clock = Callable[[], float]


@dataclass
class PipelineMetrics:
    """Collect sequential stage timings and observable cache outcomes."""

    clock: Clock = field(default=perf_counter, repr=False)
    timings_ms: dict[str, float] = field(default_factory=dict)
    cache_statuses: dict[str, str] = field(default_factory=dict)
    _started_at: float = field(init=False, repr=False)
    _checkpoint_at: float = field(init=False, repr=False)
    _finished: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        started_at = self.clock()
        self._started_at = started_at
        self._checkpoint_at = started_at

    def checkpoint(self, stage: str) -> float:
        """Close the current sequential stage and return its duration."""
        now = self.clock()
        duration_ms = max(0.0, (now - self._checkpoint_at) * 1_000)
        self.timings_ms[stage] = round(duration_ms, 3)
        self._checkpoint_at = now
        return duration_ms

    def record_cache(self, cache_name: str, status: str) -> None:
        self.cache_statuses[cache_name] = status

    def finish(self) -> float:
        """Record total elapsed time once and return it in milliseconds."""
        if not self._finished:
            now = self.clock()
            self.timings_ms["total"] = round(
                max(0.0, (now - self._started_at) * 1_000),
                3,
            )
            self._finished = True
        return self.timings_ms["total"]


def performance_status(total_ms: float, warn_after_seconds: float) -> str:
    if warn_after_seconds <= 0:
        raise ValueError("warn_after_seconds must be positive.")
    return (
        "within_budget"
        if total_ms <= warn_after_seconds * 1_000
        else "slow"
    )


def structured_event(event: str, **fields: object) -> str:
    """Return a stable JSON log payload for local logs and future collectors."""
    return json.dumps(
        {"event": event, **fields},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
