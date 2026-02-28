"""
File-level result cache for bill processing.
Cache key: (path + mtime). Re-running on the same file returns cached BillResults without OCR/LLM.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

from core.models import BillResult

logger = logging.getLogger(__name__)


def _cache_key(path: Path, mtime_ns: int) -> str:
    """Stable key for path + mtime."""
    raw = f"{path.resolve()!s}\n{mtime_ns}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _bill_result_to_dict(r: BillResult) -> dict:
    """BillResult to JSON-serializable dict."""
    return asdict(r)


def _dict_to_bill_result(d: dict) -> BillResult:
    """Dict to BillResult."""
    return BillResult(
        trace_id=d["trace_id"],
        file_name=d["file_name"],
        extraction_source=d["extraction_source"],
        structured_bill=d["structured_bill"],
        decision=d["decision"],
        policy_version_hash=d["policy_version_hash"],
        metadata=d.get("metadata", {}),
        ocr_extracted=d.get("ocr_extracted", {}),
        fusion_metadata=d.get("fusion_metadata", {}),
    )


def get_cached_results(path: Path, cache_dir: str | Path | None) -> list[BillResult] | None:
    """
    Return cached list of BillResult for this path if cache is enabled, file exists,
    and cache entry is present and valid (same mtime). Otherwise return None.
    """
    if not cache_dir or not path.exists():
        return None
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return None
    key = _cache_key(path, mtime_ns)
    cache_file = cache_dir / f"{key}.json"
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "mtime_ns" not in data or data["mtime_ns"] != mtime_ns:
            return None
        results = data.get("results", [])
        return [_dict_to_bill_result(r) for r in results]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.debug("Cache read failed for %s: %s", cache_file, e)
        return None


def set_cached_results(
    path: Path,
    results: list[BillResult],
    cache_dir: str | Path | None,
) -> None:
    """Write list of BillResult to cache for this path (using current mtime)."""
    if not cache_dir or not path.exists():
        return
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return
    key = _cache_key(path, mtime_ns)
    cache_file = cache_dir / f"{key}.json"
    data = {
        "path": str(path.resolve()),
        "mtime_ns": mtime_ns,
        "results": [_bill_result_to_dict(r) for r in results],
    }
    def _json_default(o: object) -> str:
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    try:
        cache_file.write_text(json.dumps(data, indent=0, default=_json_default), encoding="utf-8")
    except OSError as e:
        logger.debug("Cache write failed for %s: %s", cache_file, e)
