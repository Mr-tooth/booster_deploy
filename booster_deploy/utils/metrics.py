"""Shared-memory metric helpers for cross-process frequency statistics."""

from __future__ import annotations

import logging
import time
from typing import TypedDict

import numpy as np

from .synced_array import SyncedArray

logger = logging.getLogger("booster_metrics")


class MetricStats(TypedDict):
    """Represent computed timestamp-series statistics."""

    count: int
    freq_hz: float
    mean_period_s: float | None
    min_period_s: float | None
    max_period_s: float | None


class SyncedMetrics:
    """Record timestamp events in shared memory and compute period statistics."""

    def __init__(self, name: str, max_events: int = 10000) -> None:
        """Initialize a shared ring buffer for timestamp metrics.

        Args:
            name: Unique metric stream name.
            max_events: Maximum number of timestamps stored in the ring.

        """
        self.name = name
        self.max_events = int(max_events)
        self._arr = SyncedArray(
            f"metric_{name}",
            shape=(self.max_events + 2,),
            dtype="float64",
        )

    def mark(self) -> None:
        """Record one timestamp sample into the shared ring buffer."""

        def _updater(buf: np.ndarray) -> None:
            """Update ring-buffer metadata and append one timestamp."""
            write_pos = int(buf[0])
            total = int(buf[1])
            buf[2 + write_pos] = time.perf_counter()
            buf[0] = float((write_pos + 1) % self.max_events)
            buf[1] = float(total + 1)

        self._arr.modify_in_place(_updater)

    def compute(self) -> MetricStats:
        """Compute rate and period statistics from buffered timestamps.

        Returns:
            Dictionary with sample count, frequency (Hz), and period stats.

        """
        data = self._arr.read()
        write_pos = int(data[0])
        total = int(data[1])

        if total < 2:
            return {
                "count": int(total),
                "freq_hz": 0.0,
                "mean_period_s": None,
                "min_period_s": None,
                "max_period_s": None,
            }

        if total < self.max_events:
            ts = data[2 : 2 + total]
        elif write_pos == 0:
            ts = data[2 : 2 + self.max_events]
        else:
            part1 = data[2 + write_pos : 2 + self.max_events]
            part2 = data[2 : 2 + write_pos]
            ts = np.concatenate([part1, part2])

        periods = np.diff(ts)
        mean_p = float(np.mean(periods))
        return {
            "count": int(min(total, self.max_events)),
            "freq_hz": 1.0 / mean_p if mean_p > 0 else float("inf"),
            "mean_period_s": mean_p,
            "min_period_s": float(np.min(periods)),
            "max_period_s": float(np.max(periods)),
        }

    def cleanup(self) -> None:
        """Release shared resources owned by this metric recorder."""
        try:
            self._arr.cleanup()
        except Exception:
            logger.debug("SyncedMetrics cleanup failed", exc_info=True)


__all__ = ["MetricStats", "SyncedMetrics"]
