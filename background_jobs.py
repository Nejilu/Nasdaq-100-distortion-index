"""Single-worker in-process queue for stale-while-revalidate jobs."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Callable


_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="ndx-background-refresh",
)
_LOCK = Lock()
_FUTURES: dict[str, Future[object]] = {}


def submit_unique(job_key: str, function: Callable[[], object]) -> bool:
    """Submit one job per key and return whether a new job was queued."""
    with _LOCK:
        existing = _FUTURES.get(job_key)
        if existing is not None and not existing.done():
            return False
        future = _EXECUTOR.submit(function)
        _FUTURES[job_key] = future
        future.add_done_callback(lambda completed: _forget(job_key, completed))
        return True


def _forget(job_key: str, completed: Future[object]) -> None:
    with _LOCK:
        if _FUTURES.get(job_key) is completed:
            _FUTURES.pop(job_key, None)
