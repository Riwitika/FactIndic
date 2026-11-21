#!/usr/bin/env python3
"""Command-line entry point to prepare, validate, and sample FactIndic datasets.

Strict Policy:
- Default mode connects exclusively to live Hugging Face csebuetnlp/xlsum.
- Never silently falls back to fixture data.
- If live loading fails, the command stops and raises the full error.
- Fixture data requires the explicit '--fixture-test-mode-only' flag and is clearly marked as test data.

Usage (Live Experiments):
    python scripts/prepare_sample.py --language en --split test --size 5
    python scripts/prepare_sample.py --language hi --split test --size 5
    python scripts/prepare_sample.py --language all --split test --size 5

Usage (Offline Unit-Test Mode):
    python scripts/prepare_sample.py --language en --size 5 --fixture-test-mode-only
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from factindic.datasets.fixture_loader import FixtureDatasetLoader
from factindic.datasets.xlsum import (
    XLSumLoader,
    SUPPORTED_LANGUAGES,
    HF_DATASET_PARQUET_BRANCH,
    HF_DATASET_PARQUET_COMMIT_SHA,
)
from factindic.sampling import sample_records
from factindic.serialization import save_jsonl
from factindic.validation import validate_dataset


def prepare_sample_for_language(
    language: str,
    split: str,
    sample_size: int,
    seed: int,
    fixture_test_mode_only: bool,
    output_dir: Path,
    overwrite: bool,
    data_format: str = "parquet",
) -> dict:
    """Prepare and validate a dataset sample for a specific language."""
    print(f"\n{'='*70}")
    print(f"FactIndic Pipeline: Language='{language}', Split='{split}', Format='{data_format}'")
    print(f"{'='*70}")

    lang_lower = language.lower().strip()
    canonical_lang = "en" if lang_lower in ("en", "english") else "hi"

    if fixture_test_mode_only:
        print("\n[!] WARNING: RUNNING IN FIXTURE TEST MODE ONLY.")
        print("[!] Using local mock fixture data for testing purposes.")
        print("[!] THIS DATA MUST NEVER BE USED FOR RESEARCH EVALUATIONS.\n")
        fixture_filename = f"sample_{'english' if canonical_lang == 'en' else 'hindi'}.jsonl"
        fixture_path = ROOT_DIR / "data" / "fixtures" / fixture_filename
        loader = FixtureDatasetLoader(fixture_path)
        provenance = f"local_fixture:{fixture_path.name}"
        source_format = "fixture"
        branch = None
        commit_sha = None
        revision = None
    else:
        print(f"[*] Mode: STRICT LIVE DATASET LOADING")
        print(f"[*] Source: Hugging Face 'csebuetnlp/xlsum' (config='{language}', split='{split}', format='{data_format}')")
        print(f"[*] Strict Policy: No fallback to fixture or synthetic data permitted.")
        try:
            loader = XLSumLoader(language=language, data_format=data_format)
        except Exception as exc:
            print(f"[X] FATAL ERROR: Failed to initialize XLSumLoader for '{language}': {exc}", file=sys.stderr)
            raise
        source_format = data_format
        branch = HF_DATASET_PARQUET_BRANCH if data_format == "parquet" else "main"
        commit_sha = HF_DATASET_PARQUET_COMMIT_SHA if data_format == "parquet" else "30fece425f9a3866e04321773ca7a80056d55ca6"
        revision = commit_sha
        provenance = f"huggingface:csebuetnlp/xlsum:{language}:{split}:{data_format}:{commit_sha}"

    # In live mode, load the complete split to enable representative deterministic sampling across the entire split
    print(f"[*] Fetching genuine records for '{language}' (split={split})...")
    try:
        raw_records = loader.load(split=split)
    except Exception as exc:
        print(f"\n[X] STRICT NO-FALLBACK ERROR:")
        print(f"    Failed to fetch live records from Hugging Face for '{language}'.")
        print(f"    Details: {exc}\n", file=sys.stderr)
        raise

    print(f"[*] Ingested {len(raw_records)} records from {provenance}.")

    # Validate, clean, and deduplicate
    print(f"[*] Running schema validation, text normalization, and duplicate detection...")
    valid_records, invalid_records, stats = validate_dataset(
        raw_records,
        detect_duplicates=True,
        normalize_text=True,
    )

    print(f"    - Total Ingested : {stats.total_loaded}")
    print(f"    - Valid Records  : {stats.valid_count}")
    print(f"    - Invalid Records: {stats.invalid_count}")
    print(f"    - Duplicates     : {stats.duplicate_count}")
    if stats.invalid_reasons:
        print(f"    - Invalid Reasons:")
        for reason, count in stats.invalid_reasons.items():
            print(f"        * {reason}: {count}")

    if not valid_records:
        raise ValueError(f"No valid records remained after validation for language '{language}'.")

    # Deterministic reproducible sampling across the complete split using random seed
    if len(valid_records) > sample_size:
        print(f"[*] Sampling {sample_size} records across all {len(valid_records)} items with random seed={seed}...")
        sampled_records = sample_records(
            valid_records,
            sample_size=sample_size,
            seed=seed,
            shuffle=True,
        )
    else:
        sampled_records = valid_records

    print(f"[*] Final validated sample size: {len(sampled_records)} records.")

    # Save to output file
    prefix = "test_fixture" if fixture_test_mode_only else "factindic"
    output_filename = f"{prefix}_{canonical_lang}_{split}_sample_{len(sampled_records)}.jsonl"
    output_path = output_dir / output_filename
    written = save_jsonl(sampled_records, output_path, overwrite=overwrite, ensure_ascii=False)
    print(f"[*] Successfully saved {written} records to: {output_path}")

    # Save validation and audit report
    audit_data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": "csebuetnlp/xlsum",
        "language": language,
        "canonical_lang": canonical_lang,
        "split": split,
        "source_format": source_format,
        "branch": branch,
        "commit_sha": commit_sha,
        "revision": revision,
        "sample_size_requested": sample_size,
        "sample_size_obtained": len(sampled_records),
        "total_split_records_loaded": len(raw_records),
        "sampling_seed": seed,
        "provenance": provenance,
        "is_fixture_data": fixture_test_mode_only,
        "record_ids": [r.example_id for r in sampled_records],
        "statistics": stats.to_dict(),
    }

    stats_filename = f"{prefix}_{canonical_lang}_{split}_audit.json"
    stats_path = output_dir / stats_filename
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)
    print(f"[*] Saved audit report to: {stats_path}")

    return audit_data


def main():
    parser = argparse.ArgumentParser(description="FactIndic Sample Preparation Pipeline")
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language to prepare: 'en', 'hi', 'english', 'hindi', or 'all'",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split ('train', 'validation', 'test')",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=5,
        help="Number of samples to draw (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--fixture-test-mode-only",
        action="store_true",
        default=False,
        help="Explicit flag for offline unit testing ONLY using local mock fixtures.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "data" / "samples"),
        help="Destination directory for prepared samples",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="parquet",
        choices=["parquet", "archive"],
        help="Data format to use from Hugging Face: 'parquet' (default, efficient) or 'archive' (tar.bz2)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing sample files",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    languages_to_process = []
    if args.language.lower() == "all":
        languages_to_process = ["en", "hi"]
    else:
        languages_to_process = [args.language]

    for lang in languages_to_process:
        prepare_sample_for_language(
            language=lang,
            split=args.split,
            sample_size=args.size,
            seed=args.seed,
            fixture_test_mode_only=args.fixture_test_mode_only,
            output_dir=output_dir,
            overwrite=args.overwrite,
            data_format=args.format,
        )

    print(f"\n{'='*70}")
    print("[✓] FactIndic pipeline execution completed successfully.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
