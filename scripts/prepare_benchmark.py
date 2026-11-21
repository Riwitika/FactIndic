#!/usr/bin/env python3
"""Full-scale benchmark dataset preparation pipeline for FactIndic.

Prepares reproducible, high-quality English and Hindi benchmark samples from
the genuine BBC XL-Sum test split (csebuetnlp/xlsum) for the six-condition
factuality evaluation matrix.

Key Features & Methodological Safeguards:
    - Default Target: 100 English + 100 Hindi records (configurable via flags).
    - Deterministic Sampling: Reproducible selection using random seed 42.
    - Separation: English and Hindi datasets are processed and stored independently.
    - Validation: Schema validation, NFC Unicode normalization, deduplication by ID/text.
    - Content Preservation: Source article text and reference summaries are preserved intact.
    - Honest Reporting: If fewer valid records exist, reports the actual count without fabricating data.
    - Full Provenance: Generates structured audit JSON with dataset revisions, commit SHAs,
      sampling seeds, record IDs, and validation statistics.
    - Licensing Attribution: Honors CC BY-NC-SA 4.0 redistribution boundaries.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from factindic.datasets.base import BaseDatasetLoader
from factindic.datasets.fixture_loader import FixtureDatasetLoader
from factindic.datasets.xlsum import (
    HF_DATASET_PARQUET_BRANCH,
    HF_DATASET_PARQUET_COMMIT_SHA,
    XLSumLoader,
)
from factindic.sampling import sample_records
from factindic.schema import FactIndicRecord
from factindic.serialization import save_jsonl
from factindic.validation import validate_dataset

DEFAULT_EN_SIZE = 100
DEFAULT_HI_SIZE = 100
DEFAULT_SEED = 42
DEFAULT_SPLIT = "test"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "benchmark"


def prepare_language_benchmark(
    loader: BaseDatasetLoader,
    language: str,
    split: str = DEFAULT_SPLIT,
    sample_size: int = 100,
    seed: int = DEFAULT_SEED,
    output_path: Optional[Union[str, Path]] = None,
    overwrite: bool = True,
) -> Tuple[List[FactIndicRecord], Dict[str, Any]]:
    """Ingest, validate, deduplicate, and deterministically sample records for one language.

    Args:
        loader: Dataset loader instance (XLSumLoader or FixtureDatasetLoader).
        language: Language code ('en' or 'hi').
        split: Dataset split ('test', 'train', 'validation').
        sample_size: Target number of records to sample.
        seed: Random seed for deterministic sampling.
        output_path: Optional destination file path to save JSONL records.
        overwrite: Whether to overwrite existing destination files.

    Returns:
        Tuple of (list of sampled FactIndicRecord objects, dictionary of metadata and stats).
    """
    lang_lower = language.lower().strip()
    canonical_lang = "en" if lang_lower in ("en", "english") else "hi"

    # Step 1: Ingest raw records from loader
    raw_records = list(loader.load(split=split))

    # Step 2: Validate, clean, and deduplicate
    valid_records, invalid_records, stats = validate_dataset(
        raw_records,
        detect_duplicates=True,
        normalize_text=True,
    )

    # Step 3: Deterministic sampling with seed
    actual_count = len(valid_records)
    if actual_count > sample_size:
        sampled_records = sample_records(
            valid_records,
            sample_size=sample_size,
            seed=seed,
            shuffle=True,
        )
    else:
        # If fewer or equal records available, use all valid records without fabrication
        sampled_records = valid_records

    # Step 4: Save JSONL records if output path provided
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        save_jsonl(sampled_records, out_p, overwrite=overwrite, ensure_ascii=False)

    metadata: Dict[str, Any] = {
        "language": canonical_lang,
        "split": split,
        "sample_size_requested": sample_size,
        "sample_size_obtained": len(sampled_records),
        "total_records_ingested": len(raw_records),
        "valid_records_count": len(valid_records),
        "invalid_records_count": len(invalid_records),
        "duplicate_count": stats.duplicate_count,
        "sampling_seed": seed,
        "record_ids": [r.example_id for r in sampled_records],
        "statistics": stats.to_dict(),
    }

    return sampled_records, metadata


def generate_benchmark_audit(
    en_meta: Dict[str, Any],
    hi_meta: Dict[str, Any],
    seed: int = DEFAULT_SEED,
    split: str = DEFAULT_SPLIT,
    source_format: str = "parquet",
    is_fixture: bool = False,
    output_audit_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Generate and optionally save structured benchmark provenance audit report."""
    audit_data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_name": "FactIndic-Full-Scale-Benchmark",
        "dataset_name": "csebuetnlp/xlsum",
        "split": split,
        "sampling_seed": seed,
        "source_format": source_format,
        "branch": HF_DATASET_PARQUET_BRANCH if source_format == "parquet" else "main",
        "commit_sha": HF_DATASET_PARQUET_COMMIT_SHA if source_format == "parquet" else "30fece425f9a3866e04321773ca7a80056d55ca6",
        "is_fixture_data": is_fixture,
        "licensing": {
            "source_license": "CC BY-NC-SA 4.0",
            "attribution": "BBC articles via Hasan et al., Findings of ACL 2021 (XL-Sum)",
            "redistribution_policy": "Non-commercial research use only; raw full-scale benchmark text excluded from public Git tracking",
        },
        "languages": {
            "english": en_meta,
            "hindi": hi_meta,
        },
        "summary": {
            "total_requested": en_meta.get("sample_size_requested", 0) + hi_meta.get("sample_size_requested", 0),
            "total_obtained": en_meta.get("sample_size_obtained", 0) + hi_meta.get("sample_size_obtained", 0),
        },
    }

    if output_audit_path:
        aud_p = Path(output_audit_path)
        aud_p.parent.mkdir(parents=True, exist_ok=True)
        with open(aud_p, "w", encoding="utf-8") as f:
            json.dump(audit_data, f, indent=2, ensure_ascii=False)

    return audit_data


def run_benchmark_preparation(
    en_size: int = DEFAULT_EN_SIZE,
    hi_size: int = DEFAULT_HI_SIZE,
    seed: int = DEFAULT_SEED,
    split: str = DEFAULT_SPLIT,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    data_format: str = "parquet",
    fixture_test_mode_only: bool = False,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """Run full benchmark preparation for English and Hindi.

    Args:
        en_size: Target count for English records.
        hi_size: Target count for Hindi records.
        seed: Deterministic random seed.
        split: Dataset split to sample from.
        output_dir: Directory where outputs are saved.
        data_format: Loading format ('parquet' or 'archive').
        fixture_test_mode_only: If True, uses local mock fixtures for testing.
        overwrite: Whether to overwrite existing files.

    Returns:
        Audit dictionary with full provenance details.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    en_out_path = out_dir / "benchmark_en.jsonl"
    hi_out_path = out_dir / "benchmark_hi.jsonl"
    audit_out_path = out_dir / "benchmark_audit.json"

    print("=" * 70)
    print("FactIndic Full-Scale Benchmark Dataset Preparation")
    print("=" * 70)
    print(f"[*] Target Scale: {en_size} English + {hi_size} Hindi records")
    print(f"[*] Split: '{split}' | Random Seed: {seed} | Format: '{data_format}'")

    if fixture_test_mode_only:
        print("\n[!] RUNNING IN FIXTURE TEST MODE ONLY (Testing offline).")
        en_fixture = ROOT_DIR / "data" / "fixtures" / "sample_english.jsonl"
        hi_fixture = ROOT_DIR / "data" / "fixtures" / "sample_hindi.jsonl"
        en_loader = FixtureDatasetLoader(en_fixture)
        hi_loader = FixtureDatasetLoader(hi_fixture)
        source_fmt = "fixture"
    else:
        print("[*] Connecting strictly to live Hugging Face csebuetnlp/xlsum...")
        en_loader = XLSumLoader(language="english", data_format=data_format)
        hi_loader = XLSumLoader(language="hindi", data_format=data_format)
        source_fmt = data_format

    # Prepare English
    print(f"\n[*] Preparing English benchmark records (target={en_size})...")
    _, en_meta = prepare_language_benchmark(
        loader=en_loader,
        language="en",
        split=split,
        sample_size=en_size,
        seed=seed,
        output_path=en_out_path,
        overwrite=overwrite,
    )
    print(f"    [✓] Obtained {en_meta['sample_size_obtained']} valid English records.")

    # Prepare Hindi
    print(f"\n[*] Preparing Hindi benchmark records (target={hi_size})...")
    _, hi_meta = prepare_language_benchmark(
        loader=hi_loader,
        language="hi",
        split=split,
        sample_size=hi_size,
        seed=seed,
        output_path=hi_out_path,
        overwrite=overwrite,
    )
    print(f"    [✓] Obtained {hi_meta['sample_size_obtained']} valid Hindi records.")

    # Generate and save audit report
    print("\n[*] Writing provenance audit report...")
    audit_data = generate_benchmark_audit(
        en_meta=en_meta,
        hi_meta=hi_meta,
        seed=seed,
        split=split,
        source_format=source_fmt,
        is_fixture=fixture_test_mode_only,
        output_audit_path=audit_out_path,
    )
    print(f"    [✓] Audit saved to: {audit_out_path}")
    print("\n[✓] Full-Scale Benchmark Dataset Preparation Complete.")
    return audit_data


def main():
    parser = argparse.ArgumentParser(description="FactIndic Full-Scale Benchmark Preparation")
    parser.add_argument(
        "--en-size",
        type=int,
        default=DEFAULT_EN_SIZE,
        help=f"Target number of English records (default: {DEFAULT_EN_SIZE})",
    )
    parser.add_argument(
        "--hi-size",
        type=int,
        default=DEFAULT_HI_SIZE,
        help=f"Target number of Hindi records (default: {DEFAULT_HI_SIZE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Deterministic sampling seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_SPLIT,
        help=f"Dataset split to sample from (default: '{DEFAULT_SPLIT}')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--data-format",
        type=str,
        choices=["parquet", "archive"],
        default="parquet",
        help="Hugging Face data download format ('parquet' or 'archive')",
    )
    parser.add_argument(
        "--fixture-test-mode-only",
        action="store_true",
        help="Run in offline test mode using local fixtures instead of network calls",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing files in output directory",
    )

    args = parser.parse_args()

    try:
        run_benchmark_preparation(
            en_size=args.en_size,
            hi_size=args.hi_size,
            seed=args.seed,
            split=args.split,
            output_dir=args.output_dir,
            data_format=args.data_format,
            fixture_test_mode_only=args.fixture_test_mode_only,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"\n[X] Benchmark preparation failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
