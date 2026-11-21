"""Serialization and deserialization of FactIndic records to/from UTF-8 JSONL."""

import json
from pathlib import Path
from typing import List, Union

from factindic.schema import FactIndicRecord


def save_jsonl(
    records: List[FactIndicRecord],
    file_path: Union[str, Path],
    overwrite: bool = True,
    ensure_ascii: bool = False,
) -> int:
    """Export FactIndic records to a UTF-8 encoded JSONL file.

    Preserves raw Unicode characters (e.g., Devanagari script for Hindi) without
    escaping by setting ensure_ascii=False.

    Args:
        records: List of FactIndicRecord instances.
        file_path: Destination path for the .jsonl file.
        overwrite: If False, raises FileExistsError if the target file already exists.
        ensure_ascii: If False (default), writes native UTF-8 characters.

    Returns:
        The count of records written.

    Raises:
        FileExistsError: If destination exists and overwrite is False.
    """
    path = Path(file_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists and overwrite=False: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            line = json.dumps(rec.to_dict(), ensure_ascii=ensure_ascii)
            f.write(line + "\n")
            count += 1

    return count


def load_jsonl(file_path: Union[str, Path]) -> List[FactIndicRecord]:
    """Load FactIndic records from a UTF-8 encoded JSONL file.

    Args:
        file_path: Source path to the .jsonl file.

    Returns:
        List of FactIndicRecord instances.

    Raises:
        FileNotFoundError: If file does not exist.
        json.JSONDecodeError: If a line cannot be parsed as JSON.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records: List[FactIndicRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(FactIndicRecord.from_dict(data))
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(
                    f"Failed to parse line {line_num} in {path}: {exc.msg}",
                    exc.doc,
                    exc.pos,
                )

    return records
