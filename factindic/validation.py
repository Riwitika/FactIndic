"""Validation, text normalization, and dataset statistics for FactIndic."""

import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from factindic.schema import FactIndicRecord, DatasetStatistics


def clean_and_normalize_text(text: Optional[str]) -> str:
    """Safely clean and normalize text while preserving Indic scripts and Unicode integrity.

    - Applies Unicode Normalization Form C (NFC) to compose combined characters (essential
      for Indic scripts such as Devanagari matras, nuktas, and conjuncts).
    - Preserves Indic zero-width joiner (ZWJ \\u200D) and non-joiner (ZWNJ \\u200C).
    - Standardizes newline representations (\\r\\n -> \\n, \\r -> \\n).
    - Removes non-printable ASCII control characters (preserving tab and newline).
    - Strips leading and trailing whitespace.

    Args:
        text: Raw input string.

    Returns:
        Cleaned, normalized string. Returns empty string if input is None or non-string.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Unicode NFC normalization
    normalized = unicodedata.normalize("NFC", text)

    # Standardize line breaks
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # Remove non-printable control characters, but preserve tabs, newlines, and Indic ZWJ/ZWNJ
    # Keep characters in range >= 32, or \t (9), \n (10), \u200C, \u200D
    cleaned_chars = []
    for ch in normalized:
        code = ord(ch)
        if code in (9, 10, 0x200C, 0x200D) or (code >= 32 and code != 127):
            cleaned_chars.append(ch)
    cleaned = "".join(cleaned_chars)

    # Trim leading/trailing whitespace
    return cleaned.strip()


def validate_record(record: FactIndicRecord) -> Tuple[bool, Optional[str]]:
    """Validate that a FactIndicRecord meets all schema and quality criteria.

    Args:
        record: The FactIndicRecord to validate.

    Returns:
        Tuple of (is_valid, failure_reason). If valid, failure_reason is None.
    """
    if not isinstance(record, FactIndicRecord):
        return False, f"Expected FactIndicRecord instance, got {type(record).__name__}"

    if not record.example_id or not isinstance(record.example_id, str) or not record.example_id.strip():
        return False, "Missing or empty 'example_id'"

    if not record.language or not isinstance(record.language, str) or not record.language.strip():
        return False, "Missing or empty 'language'"

    if not record.source_text or not isinstance(record.source_text, str) or not record.source_text.strip():
        return False, "Missing or empty 'source_text'"

    if not record.reference_summary or not isinstance(record.reference_summary, str) or not record.reference_summary.strip():
        return False, "Missing or empty 'reference_summary'"

    # Quality check: source text must have meaningful length (at least 10 chars)
    if len(record.source_text.strip()) < 10:
        return False, f"Source text too short ({len(record.source_text.strip())} chars; min 10 required)"

    # Quality check: reference summary must have meaningful length (at least 5 chars)
    if len(record.reference_summary.strip()) < 5:
        return False, f"Reference summary too short ({len(record.reference_summary.strip())} chars; min 5 required)"

    return True, None


def compute_dataset_statistics(
    valid_records: List[FactIndicRecord],
    invalid_records: List[Tuple[Any, str]],
    duplicate_count: int,
    total_loaded: int,
) -> DatasetStatistics:
    """Compute comprehensive dataset statistics.

    Args:
        valid_records: List of successfully validated FactIndicRecords.
        invalid_records: List of tuples containing (record/data, failure_reason).
        duplicate_count: Number of duplicate records detected.
        total_loaded: Total number of raw records ingested.

    Returns:
        DatasetStatistics instance with full metrics.
    """
    lang_dist = Counter(rec.language for rec in valid_records)
    reasons_dist = Counter(reason for _, reason in invalid_records)

    total_source_chars = sum(len(rec.source_text) for rec in valid_records)
    total_summary_chars = sum(len(rec.reference_summary) for rec in valid_records)
    v_count = len(valid_records)

    avg_src = round(total_source_chars / v_count, 2) if v_count > 0 else 0.0
    avg_sum = round(total_summary_chars / v_count, 2) if v_count > 0 else 0.0

    return DatasetStatistics(
        total_loaded=total_loaded,
        valid_count=v_count,
        invalid_count=len(invalid_records),
        duplicate_count=duplicate_count,
        language_distribution=dict(lang_dist),
        invalid_reasons=dict(reasons_dist),
        average_source_length_chars=avg_src,
        average_summary_length_chars=avg_sum,
    )


def validate_dataset(
    records: List[FactIndicRecord],
    detect_duplicates: bool = True,
    normalize_text: bool = True,
) -> Tuple[List[FactIndicRecord], List[Tuple[FactIndicRecord, str]], DatasetStatistics]:
    """Validate a collection of FactIndicRecords, handling deduplication and normalization.

    Deduplication checks exact text match of (source_text, reference_summary) and example_id.
    Distinct articles with different texts are preserved even if they share metadata or source domains.

    Args:
        records: Collection of records to validate.
        detect_duplicates: Whether to detect and isolate duplicate records.
        normalize_text: Whether to apply clean_and_normalize_text to text fields.

    Returns:
        Tuple of:
          - valid_records: List of validated, unique FactIndicRecord objects.
          - invalid_records: List of tuples (record, reason) for any rejected items.
          - statistics: DatasetStatistics object summarizing the dataset.
    """
    valid_records: List[FactIndicRecord] = []
    invalid_records: List[Tuple[FactIndicRecord, str]] = []

    seen_ids: Set[str] = set()
    seen_content_hashes: Set[Tuple[str, str]] = set()
    duplicate_count = 0
    total_loaded = len(records)

    for rec in records:
        if normalize_text:
            rec = FactIndicRecord(
                example_id=rec.example_id.strip() if rec.example_id else "",
                language=rec.language.strip() if rec.language else "",
                source_text=clean_and_normalize_text(rec.source_text),
                reference_summary=clean_and_normalize_text(rec.reference_summary),
                metadata=rec.metadata,
            )

        is_valid, reason = validate_record(rec)
        if not is_valid:
            invalid_records.append((rec, str(reason)))
            continue

        if detect_duplicates:
            # Check duplicate ID
            if rec.example_id in seen_ids:
                duplicate_count += 1
                invalid_records.append((rec, f"Duplicate example_id: '{rec.example_id}'"))
                continue

            # Check duplicate content (identical source and reference summary)
            content_key = (rec.source_text, rec.reference_summary)
            if content_key in seen_content_hashes:
                duplicate_count += 1
                invalid_records.append((rec, f"Duplicate content identical to existing record"))
                continue

            seen_ids.add(rec.example_id)
            seen_content_hashes.add(content_key)

        valid_records.append(rec)

    stats = compute_dataset_statistics(
        valid_records=valid_records,
        invalid_records=invalid_records,
        duplicate_count=duplicate_count,
        total_loaded=total_loaded,
    )

    return valid_records, invalid_records, stats
