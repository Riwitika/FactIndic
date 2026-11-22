"""Benchmark Manifest and Cryptographic Integrity Verification.

Provides deterministic SHA-256 fingerprinting and structure validation for full-scale
FactIndic benchmark datasets (English and Hindi XL-Sum test splits).
Ensures scientific reproducibility and data integrity without committing raw text.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def compute_file_sha256(filepath: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        filepath: Path to the target file.
        chunk_size: Block size in bytes for reading.

    Returns:
        Hex-encoded SHA-256 string.
    """
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_benchmark_manifest(
    benchmark_dir: Union[str, Path],
    en_filename: str = "benchmark_en.jsonl",
    hi_filename: str = "benchmark_hi.jsonl",
    seed: int = 42,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Generate a machine-readable manifest of benchmark files with cryptographic checksums.

    Args:
        benchmark_dir: Directory containing benchmark dataset files.
        en_filename: Filename for English benchmark.
        hi_filename: Filename for Hindi benchmark.
        seed: Sampling seed used to select records.
        output_path: Optional path to write manifest JSON file.

    Returns:
        Dictionary representation of the benchmark manifest.
    """
    b_dir = Path(benchmark_dir)
    en_path = b_dir / en_filename
    hi_path = b_dir / hi_filename

    if not en_path.exists():
        raise FileNotFoundError(f"English benchmark file not found: {en_path}")
    if not hi_path.exists():
        raise FileNotFoundError(f"Hindi benchmark file not found: {hi_path}")

    files_info: Dict[str, Any] = {}
    total_records = 0
    all_record_ids: List[str] = []

    for lang, p, fname in [("en", en_path, en_filename), ("hi", hi_path, hi_filename)]:
        sha256_hash = compute_file_sha256(p)
        record_ids: List[str] = []
        with open(p, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                record = json.loads(s)
                r_id = str(record.get("example_id") or record.get("id") or f"{lang}_{line_no}")
                record_ids.append(r_id)

        files_info[lang] = {
            "filename": fname,
            "sha256": sha256_hash,
            "record_count": len(record_ids),
            "record_ids": record_ids,
        }
        total_records += len(record_ids)
        all_record_ids.extend(record_ids)

    unique_ids = set(all_record_ids)
    has_duplicate_ids = len(unique_ids) < len(all_record_ids)

    manifest: Dict[str, Any] = {
        "manifest_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "XL-Sum",
        "source_split": "test",
        "xlsum_repo_commit": "ab0484252a17cb12999710f4ccdd9966141a2fa3",
        "sampling_seed": seed,
        "total_records": total_records,
        "languages": files_info,
        "integrity": {
            "no_duplicate_ids": not has_duplicate_ids,
            "balanced_splits": files_info["en"]["record_count"] == files_info["hi"]["record_count"],
        },
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def validate_benchmark_manifest(
    manifest_path: Union[str, Path],
    benchmark_dir: Optional[Union[str, Path]] = None,
) -> Tuple[bool, List[str]]:
    """Validate that benchmark files on disk match the manifest checksums and counts.

    Args:
        manifest_path: Path to the benchmark_manifest.json.
        benchmark_dir: Optional directory override. Defaults to manifest's parent directory.

    Returns:
        Tuple of (is_valid: bool, errors: List[str]).
    """
    m_path = Path(manifest_path)
    if not m_path.exists():
        return False, [f"Manifest file not found: {m_path}"]

    try:
        with open(m_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as exc:
        return False, [f"Failed to parse manifest JSON: {exc}"]

    errors: List[str] = []
    b_dir = Path(benchmark_dir) if benchmark_dir else m_path.parent
    languages = manifest.get("languages", {})
    if not languages:
        errors.append("Manifest contains no language file definitions.")
        return False, errors

    all_seen_ids: Dict[str, str] = {}  # id -> lang

    for lang, meta in languages.items():
        fname = meta.get("filename")
        expected_sha = meta.get("sha256")
        expected_count = meta.get("record_count")
        expected_ids = meta.get("record_ids", [])

        if not fname:
            errors.append(f"Missing filename for language '{lang}'.")
            continue

        file_p = b_dir / fname
        if not file_p.exists():
            errors.append(f"File for language '{lang}' does not exist: {file_p}")
            continue

        actual_sha = compute_file_sha256(file_p)
        if expected_sha and actual_sha != expected_sha:
            errors.append(
                f"SHA-256 mismatch for '{fname}': expected {expected_sha}, found {actual_sha}."
            )

        actual_ids: List[str] = []
        with open(file_p, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except Exception as exc:
                    errors.append(f"{fname}:{line_idx}: Invalid JSON record: {exc}")
                    continue

                r_id = str(rec.get("example_id") or rec.get("id") or "")
                if not r_id:
                    errors.append(f"{fname}:{line_idx}: Missing 'example_id' or 'id' field.")
                elif r_id in all_seen_ids:
                    errors.append(
                        f"Duplicate record ID '{r_id}' found in {fname}:{line_idx} (previously seen in {all_seen_ids[r_id]})."
                    )
                else:
                    all_seen_ids[r_id] = fname
                actual_ids.append(r_id)

                # Validate source article and summary
                text = str(rec.get("source_text") or rec.get("text", "")).strip()
                summary = str(rec.get("reference_summary") or rec.get("summary", "")).strip()
                if not text:
                    errors.append(f"{fname}:{line_idx} (id={r_id}): Empty source text.")
                if not summary:
                    errors.append(f"{fname}:{line_idx} (id={r_id}): Empty reference summary.")

        if expected_count is not None and len(actual_ids) != expected_count:
            errors.append(
                f"Record count mismatch for '{fname}': expected {expected_count}, found {len(actual_ids)}."
            )

        if expected_ids and set(expected_ids) != set(actual_ids):
            missing_ids = set(expected_ids) - set(actual_ids)
            extra_ids = set(actual_ids) - set(expected_ids)
            if missing_ids:
                errors.append(f"'{fname}' is missing expected IDs: {list(missing_ids)[:5]}...")
            if extra_ids:
                errors.append(f"'{fname}' has unexpected IDs: {list(extra_ids)[:5]}...")

    is_valid = len(errors) == 0
    return is_valid, errors
