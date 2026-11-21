#!/usr/bin/env python3
"""Strict Live XL-Sum Verification Script for FactIndic (Phase 1.5).

Verifies live retrieval of genuine records from Hugging Face dataset 'csebuetnlp/xlsum'
for both English and Hindi.

Strict Policy:
- Never falls back to fixtures or synthetic data.
- Halts and reports full error details if live fetching fails.
- Saves audit report to data/audits/live_verification_audit.json.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from factindic.datasets.xlsum import XLSumLoader
from factindic.schema import FactIndicRecord
from factindic.validation import clean_and_normalize_text, validate_record


def verify_language(
    language: str,
    split: str = "test",
    target_count: int = 2,
    timeout: int = 300,
) -> Dict[str, Any]:
    """Verify live XL-Sum loading for a specific language configuration."""
    start_time = time.time()
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    print(f"\n{'='*75}")
    print(f"[LIVE VERIFICATION] Dataset: csebuetnlp/xlsum | Language: '{language}' | Split: '{split}'")
    print(f"{'='*75}")

    audit_entry: Dict[str, Any] = {
        "dataset_repository": "csebuetnlp/xlsum",
        "language_requested": language,
        "split_requested": split,
        "sample_size_requested": target_count,
        "retrieval_timestamp_utc": timestamp_utc,
        "verification_status": "FAILED",
        "error": None,
        "records_retrieved_count": 0,
        "returned_field_names": [],
        "unique_record_id_count": 0,
        "valid_count": 0,
        "invalid_count": 0,
        "records_audit": [],
    }

    try:
        loader = XLSumLoader(language=language, timeout=timeout)
        audit_entry["hf_config"] = loader.hf_config
        audit_entry["lang_alias"] = loader.lang_alias

        print(f"[*] Initialized XLSumLoader for configuration '{loader.hf_config}' (alias: '{loader.lang_alias}')")
        print(f"[*] Connecting to live Hugging Face repository stream (split='{split}', count={target_count})...")

        raw_records: List[Dict[str, Any]] = []
        mapped_records: List[FactIndicRecord] = []

        for raw_item, record in loader.stream_raw(split=split, max_samples=target_count):
            raw_records.append(raw_item)
            mapped_records.append(record)

        audit_entry["records_retrieved_count"] = len(raw_records)

        if len(raw_records) == 0:
            raise ValueError(f"No records were yielded from live stream for '{language}' ({split}).")

        # Verify fields returned by genuine Hugging Face dataset
        returned_fields = sorted(list(raw_records[0].keys()))
        audit_entry["returned_field_names"] = returned_fields
        print(f"[✓] Successfully retrieved {len(raw_records)} genuine records.")
        print(f"[*] Genuine returned fields: {returned_fields}")

        # Check required fields
        required_fields = ["id", "url", "title", "summary", "text"]
        missing_fields = [f for f in required_fields if f not in returned_fields]
        if missing_fields:
            raise KeyError(f"Missing expected fields in live data: {missing_fields}")

        # Verify and validate each record
        seen_ids = set()
        valid_count = 0
        invalid_count = 0

        for idx, (raw_item, record) in enumerate(zip(raw_records, mapped_records), 1):
            raw_id = raw_item.get("id")
            title = raw_item.get("title", "")
            raw_text = raw_item.get("text", "")
            raw_summary = raw_item.get("summary", "")

            # Check nonempty
            if not raw_text or not raw_text.strip():
                raise ValueError(f"Record {idx} (ID: {raw_id}) has empty source text!")
            if not raw_summary or not raw_summary.strip():
                raise ValueError(f"Record {idx} (ID: {raw_id}) has empty reference summary!")

            seen_ids.add(raw_id)

            # Validate against FactIndicRecord schema
            is_valid, reason = validate_record(record)
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1

            record_audit = {
                "index": idx,
                "raw_id": raw_id,
                "example_id": record.example_id,
                "language": record.language,
                "title": title[:80],
                "url": raw_item.get("url"),
                "source_text_length_chars": len(raw_text),
                "summary_length_chars": len(raw_summary),
                "is_schema_valid": is_valid,
                "validation_reason": reason,
            }
            audit_entry["records_audit"].append(record_audit)

            print(f"\n  Record #{idx}:")
            print(f"    - Genuine ID     : {raw_id}")
            print(f"    - Example ID     : {record.example_id}")
            print(f"    - Language       : {record.language}")
            print(f"    - Title          : {title[:70]}")
            print(f"    - URL            : {raw_item.get('url')}")
            print(f"    - Source Chars   : {len(raw_text)}")
            print(f"    - Summary Chars  : {len(raw_summary)}")
            print(f"    - Schema Valid   : {is_valid}")

        audit_entry["unique_record_id_count"] = len(seen_ids)
        audit_entry["valid_count"] = valid_count
        audit_entry["invalid_count"] = invalid_count

        # Check for duplicates within sample
        if len(seen_ids) != len(raw_records):
            raise ValueError(f"Duplicate IDs found in retrieved sample: {len(raw_records) - len(seen_ids)}")

        elapsed = round(time.time() - start_time, 2)
        audit_entry["elapsed_seconds"] = elapsed
        audit_entry["verification_status"] = "PASSED"
        print(f"\n[✓] VERIFICATION PASSED for '{language}' in {elapsed}s.")

    except Exception as exc:
        elapsed = round(time.time() - start_time, 2)
        audit_entry["elapsed_seconds"] = elapsed
        audit_entry["verification_status"] = "FAILED"
        audit_entry["error"] = str(exc)
        print(f"\n[X] VERIFICATION FAILED for '{language}': {exc}", file=sys.stderr)

    return audit_entry


def main():
    parser = argparse.ArgumentParser(description="Strict Live XL-Sum Verification")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["english", "hindi"],
        help="Languages to verify (default: 'english' 'hindi')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: 'test')",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=2,
        help="Number of records to verify per language (default: 2)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds per language (default: 300)",
    )
    parser.add_argument(
        "--audit-dir",
        type=str,
        default=str(ROOT_DIR / "data" / "audits"),
        help="Destination directory for verification audit report",
    )

    args = parser.parse_args()
    audit_dir = Path(args.audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)

    overall_results: Dict[str, Any] = {
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": "FactIndic Phase 1.5 Strict Live XL-Sum Verification",
        "dataset": "csebuetnlp/xlsum",
        "split": args.split,
        "sample_size_per_language": args.count,
        "languages": {},
        "all_passed": False,
    }

    all_passed = True
    for lang in args.languages:
        result = verify_language(
            language=lang,
            split=args.split,
            target_count=args.count,
            timeout=args.timeout,
        )
        overall_results["languages"][lang] = result
        if result["verification_status"] != "PASSED":
            all_passed = False

    overall_results["all_passed"] = all_passed

    # Save summary audit report
    audit_file = audit_dir / "live_xlsum_verification_audit.json"
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*75}")
    print(f"[*] Audit report successfully written to: {audit_file}")
    print(f"[*] Overall Live Verification Result: {'ALL PASSED [✓]' if all_passed else 'SOME FAILED [X]'}")
    print(f"{'='*75}\n")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
