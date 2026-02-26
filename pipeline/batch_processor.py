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
        file_paths: list[str | Path] | list[tuple[str, str | Path]] | list[tuple[str, str | Path, str, str]],
        *,
        stop_on_first_error: bool = False,
    ) -> tuple[list[BillResult], BatchMetrics] | tuple[list[tuple[str, BillResult]], BatchMetrics]:
        """
        Run pipeline.process() for each file. Returns (results, metrics).
        If file_paths is list of (source_folder, path) or (source_folder, path, expense_type, employee_id),
        returns (list of (source_folder, BillResult), metrics). Otherwise returns (list of BillResult, metrics).
        On error: if stop_on_first_error, re-raise; else log and continue, increment failed_count.
        """
        with_folders = bool(
            file_paths
            and isinstance(file_paths[0], (list, tuple))
            and len(file_paths[0]) >= 2
        )
        results: list[BillResult] = []
        results_with_folders: list[tuple[str, BillResult]] = []
        metrics = BatchMetrics()
        start = time.perf_counter()
        for item in file_paths:
            if with_folders:
                source_folder, path = item[0], item[1]
                expense_type = item[2] if len(item) >= 4 else None
                employee_id = item[3] if len(item) >= 4 else None
            else:
                source_folder, path = "", item
                expense_type, employee_id = None, None
            p = Path(path)
            if not p.exists():
                logger.warning("Skip missing file: %s", p)
                metrics.failed_count += 1
                continue
            try:
                if p.suffix.lower() == ".pdf":
                    page_results = self._pipeline.process_multi(
                        p, expense_type=expense_type, employee_id=employee_id
                    )
                else:
                    page_results = self._pipeline.process(
                        p, expense_type=expense_type, employee_id=employee_id
                    )
                for result in page_results:
                    results.append(result)
                    if with_folders:
                        results_with_folders.append((source_folder, result))
                    _update_metrics(metrics, result)
            except Exception as e:
                metrics.failed_count += 1
                logger.exception("Batch item failed %s: %s", p, e)
                if stop_on_first_error:
                    raise
        metrics.total_time_sec = time.perf_counter() - start
        if with_folders:
            return results_with_folders, metrics
        return results, metrics
