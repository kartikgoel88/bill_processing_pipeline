"""
Batch processor: list of files -> run pipeline per file, collect metrics.
Does not duplicate pipeline logic; uses BillProcessingPipeline.process().
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from core.models import BillResult, BatchMetrics
from pipeline.bill_pipeline import BillProcessingPipeline

logger = logging.getLogger(__name__)


def _update_metrics(metrics: BatchMetrics, result: BillResult) -> None:
    """Update counts from a single BillResult."""
    metrics.total_processed += 1
    decision = (result.decision.get("decision") or "").upper()
    if decision == "APPROVED":
        metrics.approved_count += 1
    elif decision == "REJECTED":
        metrics.rejected_count += 1
    elif decision == "NEEDS_REVIEW":
        metrics.needs_review_count += 1


class BatchProcessor:
    """
    Process multiple files sequentially (or async later). Collects metrics.
    Injected pipeline; no duplicate pipeline logic.
    """

    def __init__(self, pipeline: BillProcessingPipeline) -> None:
        self._pipeline = pipeline

    def process_batch(
        self,
        file_paths: list[str | Path],
        *,
        stop_on_first_error: bool = False,
    ) -> tuple[list[BillResult], BatchMetrics]:
        """
        Run pipeline.process() for each file. Returns (results, metrics).
        On error: if stop_on_first_error, re-raise; else log and continue, increment failed_count.
        """
        results: list[BillResult] = []
        metrics = BatchMetrics()
        start = time.perf_counter()
        for path in file_paths:
            p = Path(path)
            if not p.exists():
                logger.warning("Skip missing file: %s", p)
                metrics.failed_count += 1
                continue
            try:
                result = self._pipeline.process(p)
                results.append(result)
                _update_metrics(metrics, result)
            except Exception as e:
                metrics.failed_count += 1
                logger.exception("Batch item failed %s: %s", p, e)
                if stop_on_first_error:
                    raise
        metrics.total_time_sec = time.perf_counter() - start
        return results, metrics
