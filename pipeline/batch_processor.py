"""
Batch processor: list of files -> run pipeline per file, collect metrics.
Does not duplicate pipeline logic; uses BillProcessingPipeline.process().
Supports parallel execution via max_workers (ThreadPoolExecutor).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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


def _process_one_indexed(
    index: int,
    pipeline: BillProcessingPipeline,
    item: tuple[str, str | Path] | tuple[str, str | Path, str, str] | str | Path,
    with_folders: bool,
    stop_on_first_error: bool,
) -> tuple[int, list[BillResult], list[tuple[str, BillResult]], int, Exception | None]:
    """Process one file. Returns (index, results, results_with_folders, failed_count, error)."""
    if with_folders:
        source_folder, path = item[0], item[1]
        expense_type = item[2] if len(item) >= 4 else None
        employee_id = item[3] if len(item) >= 4 else None
    else:
        source_folder, path = "", item
        expense_type, employee_id = None, None
    p = Path(path)
    folder_label = source_folder if source_folder else "."
    if not p.exists():
        logger.warning("Skip missing file: folder=%s file=%s", folder_label, p)
        return (index, [], [], 1, None)
    logger.info("Processing folder=%s file=%s", folder_label, p.name)
    try:
        if p.suffix.lower() == ".pdf":
            page_results = pipeline.process_multi(
                p, expense_type=expense_type, employee_id=employee_id
            )
        else:
            page_results = pipeline.process(
                p, expense_type=expense_type, employee_id=employee_id
            )
        results_with_folders: list[tuple[str, BillResult]] = [
            (source_folder, r) for r in page_results
        ] if with_folders else []
        return (index, page_results, results_with_folders, 0, None)
    except Exception as e:
        logger.exception("Batch item failed folder=%s file=%s: %s", folder_label, p.name, e)
        return (index, [], [], 1, e)


class BatchProcessor:
    """
    Process multiple files in parallel (or sequentially when max_workers=1). Collects metrics.
    Injected pipeline; no duplicate pipeline logic.
    """

    def __init__(self, pipeline: BillProcessingPipeline, max_workers: int = 1) -> None:
        self._pipeline = pipeline
        self._max_workers = max(1, int(max_workers))

    def process_batch(
        self,
        file_paths: list[str | Path] | list[tuple[str, str | Path]] | list[tuple[str, str | Path, str, str]],
        *,
        stop_on_first_error: bool = False,
    ) -> tuple[list[BillResult], BatchMetrics] | tuple[list[tuple[str, BillResult]], BatchMetrics]:
        """
        Run pipeline.process() for each file (in parallel when max_workers > 1). Returns (results, metrics).
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

        if self._max_workers > 1:
            # Parallel
            completed_by_index: dict[int, tuple[list[BillResult], list[tuple[str, BillResult]], int, Exception | None]] = {}
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = {
                    executor.submit(
                        _process_one_indexed,
                        i,
                        self._pipeline,
                        item,
                        with_folders,
                        stop_on_first_error,
                    ): i
                    for i, item in enumerate(file_paths)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        _index, res, res_f, failed, err = future.result()
                        completed_by_index[idx] = (res, res_f, failed, err)
                        if err and stop_on_first_error:
                            raise err
                    except Exception as e:
                        completed_by_index[idx] = ([], [], 1, e)
                        if stop_on_first_error:
                            raise
            for i in range(len(file_paths)):
                res, res_f, failed, _err = completed_by_index.get(i, ([], [], 1, None))
                metrics.failed_count += failed
                for r in res:
                    results.append(r)
                    _update_metrics(metrics, r)
                if with_folders:
                    results_with_folders.extend(res_f)
        else:
            # Sequential (max_workers <= 1)
            for item in file_paths:
                if with_folders:
                    source_folder, path = item[0], item[1]
                    expense_type = item[2] if len(item) >= 4 else None
                    employee_id = item[3] if len(item) >= 4 else None
                else:
                    source_folder, path = "", item
                    expense_type, employee_id = None, None
                p = Path(path)
                folder_label = source_folder if source_folder else "."
                if not p.exists():
                    logger.warning("Skip missing file: folder=%s file=%s", folder_label, p)
                    metrics.failed_count += 1
                    continue
                logger.info("Processing folder=%s file=%s", folder_label, p.name)
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
                    logger.exception("Batch item failed folder=%s file=%s: %s", folder_label, p.name, e)
                    if stop_on_first_error:
                        raise
        metrics.total_time_sec = time.perf_counter() - start
        if with_folders:
            return results_with_folders, metrics
        return results, metrics
