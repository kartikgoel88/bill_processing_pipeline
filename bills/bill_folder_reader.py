"""
Bill folder reader: traverses input root with structure expense_type/employee_id/bills.
Yields (expense_type, employee_id, file_path) for each bill file.
Infers expense_type from folder name and employee_id from subfolder.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

BILL_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


def iter_bills(root_folder: str | Path) -> Iterator[tuple[str, str, Path]]:
    """
    Traverse root_folder and yield (expense_type, employee_id, file_path) for each bill.

    Expected structure:
        root/
            fuel/
            meal/
            commute/
                EMP001/
                EMP002/
                    bill1.pdf
                    bill2.jpg

    Also supports single expense_type root: root/employee_id/files (expense_type = root name).
    """
    root = Path(root_folder).resolve()
    if not root.is_dir():
        logger.warning("Input root is not a directory: %s", root)
        return

    # Check if this is a single expense_type folder (root/employee_id/files)
    first_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if first_dirs:
        first_sub = first_dirs[0]
        sub_items = list(first_sub.iterdir())
        has_subdirs = any(p.is_dir() for p in sub_items)
        if not has_subdirs and any(
            p.is_file() and p.suffix.lower() in BILL_EXTENSIONS for p in sub_items
        ):
            expense_type = root.name.strip().lower()
            for emp_dir in first_dirs:
                employee_id = emp_dir.name.strip()
                if not employee_id:
                    continue
                for file_path in sorted(emp_dir.iterdir()):
                    if not file_path.is_file() or file_path.suffix.lower() not in BILL_EXTENSIONS:
                        continue
                    yield (expense_type, employee_id, file_path)
            return

    # Full structure: root/expense_type/employee_id/files
    for expense_dir in sorted(root.iterdir()):
        if not expense_dir.is_dir():
            continue
        expense_type = expense_dir.name.strip().lower()
        if not expense_type:
            continue
        for emp_dir in sorted(expense_dir.iterdir()):
            if not emp_dir.is_dir():
                continue
            employee_id = emp_dir.name.strip()
            if not employee_id:
                continue
            for file_path in sorted(emp_dir.iterdir()):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in BILL_EXTENSIONS:
                    continue
                yield (expense_type, employee_id, file_path)
