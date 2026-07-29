import json
import logging

import pytest

from observability import (
    PipelineMetrics,
    performance_status,
    structured_event,
)


class _Clock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def test_pipeline_metrics_records_sequential_stages_and_total():
    metrics = PipelineMetrics(clock=_Clock(10.0, 10.025, 10.075, 10.1))

    metrics.checkpoint("holdings")
    metrics.checkpoint("market_data")
    metrics.record_cache("nasdaq_universe", "fallback_hit")
    total_ms = metrics.finish()

    assert metrics.timings_ms == {
        "holdings": 25.0,
        "market_data": 50.0,
        "total": 100.0,
    }
    assert metrics.cache_statuses == {"nasdaq_universe": "fallback_hit"}
    assert total_ms == 100.0


def test_pipeline_finish_is_idempotent():
    metrics = PipelineMetrics(clock=_Clock(1.0, 1.25))

    assert metrics.finish() == 250.0
    assert metrics.finish() == 250.0


@pytest.mark.parametrize(
    ("total_ms", "warn_after_seconds", "expected"),
    [
        (999.0, 1.0, "within_budget"),
        (1_000.0, 1.0, "within_budget"),
        (1_001.0, 1.0, "slow"),
    ],
)
def test_performance_status(total_ms, warn_after_seconds, expected):
    assert performance_status(total_ms, warn_after_seconds) == expected


def test_performance_status_rejects_non_positive_budget():
    with pytest.raises(ValueError, match="positive"):
        performance_status(10.0, 0)


def test_structured_event_is_machine_readable(caplog):
    caplog.set_level(logging.INFO)
    payload = structured_event(
        "snapshot_recompute_complete",
        snapshot_id=42,
        timings_ms={"total": 123.4},
    )
    logging.getLogger("test").info(payload)

    decoded = json.loads(caplog.messages[-1])
    assert decoded["event"] == "snapshot_recompute_complete"
    assert decoded["snapshot_id"] == 42
    assert decoded["timings_ms"]["total"] == 123.4
