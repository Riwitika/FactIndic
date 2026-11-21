#!/usr/bin/env python3
"""CLI script to execute the six-condition experiment on prepared benchmarks.

Supports full-scale benchmark evaluation, preflight checks, incremental resumption,
and structured audit reporting.

Usage:
    python scripts/run_benchmark_experiment.py \
        --en-benchmark data/benchmark/benchmark_en.jsonl \
        --hi-benchmark data/benchmark/benchmark_hi.jsonl \
        --output-dir data/benchmark_results \
        --delay 0.5
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from factindic.experiment_runner import (
    ConditionID,
    ExperimentRecord,
    UnifiedExperimentRunner,
    load_experiment_records,
    save_experiment_records,
)
from factindic.models.aya import CohereClient
from factindic.schema import FactIndicRecord
from factindic.serialization import load_jsonl


def preflight_check(
    en_path: Path,
    hi_path: Path,
    client: Optional[CohereClient] = None,
) -> Dict[str, Any]:
    """Validate benchmark files, record counts, and API access prior to execution.

    Args:
        en_path: Path to English benchmark JSONL.
        hi_path: Path to Hindi benchmark JSONL.
        client: Optional configured CohereClient.

    Returns:
        Structured dictionary with preflight status, counts, and blockers.
    """
    blockers: List[str] = []

    # 1. Benchmark file existence
    if not en_path.exists():
        blockers.append(f"English benchmark file missing: {en_path}")
    if not hi_path.exists():
        blockers.append(f"Hindi benchmark file missing: {hi_path}")

    en_records: List[FactIndicRecord] = []
    hi_records: List[FactIndicRecord] = []

    if en_path.exists():
        try:
            en_records = load_jsonl(en_path)
            if not en_records:
                blockers.append(f"English benchmark file is empty: {en_path}")
        except Exception as e:
            blockers.append(f"Failed to load English benchmark: {e}")

    if hi_path.exists():
        try:
            hi_records = load_jsonl(hi_path)
            if not hi_records:
                blockers.append(f"Hindi benchmark file is empty: {hi_path}")
        except Exception as e:
            blockers.append(f"Failed to load Hindi benchmark: {e}")

    # 2. Check credentials
    llm = client or CohereClient()
    if not llm.api_key:
        blockers.append("COHERE_API_KEY is not configured in environment or .env file.")

    total_records = len(en_records) + len(hi_records)
    # 2 prompt generations per record + ~2 claim extraction calls per Prompt A summary + ~2 claim extraction calls per Prompt B summary
    estimated_generation_calls = total_records * 2
    estimated_extraction_calls = total_records * 4  # Average 2 sentences per summary * 2 summaries
    estimated_total_api_calls = estimated_generation_calls + estimated_extraction_calls

    return {
        "ready": len(blockers) == 0,
        "blockers": blockers,
        "en_count": len(en_records),
        "hi_count": len(hi_records),
        "total_source_records": total_records,
        "total_condition_evaluations": total_records * 6,
        "estimated_generation_calls": estimated_generation_calls,
        "estimated_extraction_calls": estimated_extraction_calls,
        "estimated_total_api_calls": estimated_total_api_calls,
        "model_id": llm.model_id,
        "provider": llm.provider,
    }


def execute_benchmark_experiment(
    en_benchmark_path: Union[str, Path],
    hi_benchmark_path: Union[str, Path],
    output_dir: Union[str, Path],
    runner: Optional[UnifiedExperimentRunner] = None,
    delay_seconds: float = 0.5,
    resume: bool = True,
) -> Dict[str, Any]:
    """Execute the six-condition experiment across English and Hindi benchmarks.

    Args:
        en_benchmark_path: File path to benchmark_en.jsonl.
        hi_benchmark_path: File path to benchmark_hi.jsonl.
        output_dir: Directory to save results and audit report.
        runner: Optional pre-configured UnifiedExperimentRunner.
        delay_seconds: Delay between API calls.
        resume: Whether to resume and skip completed conditions.

    Returns:
        Audit report dictionary.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_file = out_dir / "experiment_results.jsonl"
    audit_file = out_dir / "experiment_audit.json"

    en_path = Path(en_benchmark_path)
    hi_path = Path(hi_benchmark_path)

    # 1. Run Preflight
    exp_runner = runner or UnifiedExperimentRunner(delay_seconds=delay_seconds)
    preflight = preflight_check(en_path, hi_path, client=exp_runner.llm_client)
    if not preflight["ready"]:
        raise RuntimeError(f"Preflight check failed: {'; '.join(preflight['blockers'])}")

    en_records = load_jsonl(en_path)
    hi_records = load_jsonl(hi_path)
    all_records = en_records + hi_records

    t_start = time.time()
    start_time_utc = datetime.now(timezone.utc).isoformat()

    # 2. Run all six conditions with incremental persistence
    results = exp_runner.run_all_conditions(
        records=all_records,
        output_path=res_file,
        resume=resume,
    )

    t_elapsed = round(time.time() - t_start, 2)

    # 3. Compile validation statistics
    condition_counts: Dict[str, int] = {}
    success_counts: Dict[str, int] = {}
    failed_counts: Dict[str, int] = {}
    lang_counts: Dict[str, int] = {"en": 0, "hi": 0}

    for r in results:
        cid = r.condition_id
        condition_counts[cid] = condition_counts.get(cid, 0) + 1
        if r.status == "SUCCESS":
            success_counts[cid] = success_counts.get(cid, 0) + 1
        else:
            failed_counts[cid] = failed_counts.get(cid, 0) + 1
        lang_counts[r.language] = lang_counts.get(r.language, 0) + 1

    audit_report = {
        "timestamp_utc": start_time_utc,
        "completed_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": t_elapsed,
        "experiment_name": "FactIndic-Six-Condition-Benchmark-Experiment",
        "model_id": exp_runner.llm_client.model_id,
        "provider": exp_runner.llm_client.provider,
        "generation_parameters": exp_runner.generation_params,
        "source_records": {
            "english_count": len(en_records),
            "hindi_count": len(hi_records),
            "total_count": len(all_records),
        },
        "results_summary": {
            "total_evaluations": len(results),
            "expected_evaluations": len(all_records) * 6,
            "condition_breakdown": condition_counts,
            "success_breakdown": success_counts,
            "failed_breakdown": failed_counts,
            "language_distribution": lang_counts,
        },
        "output_files": {
            "results_jsonl": str(res_file.name),
            "audit_json": str(audit_file.name),
        },
    }

    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2, ensure_ascii=False)

    return audit_report


def main():
    parser = argparse.ArgumentParser(description="FactIndic Full-Scale Benchmark Experiment Runner")
    parser.add_argument(
        "--en-benchmark",
        type=str,
        default="data/benchmark/benchmark_en.jsonl",
        help="Path to English benchmark JSONL",
    )
    parser.add_argument(
        "--hi-benchmark",
        type=str,
        default="data/benchmark/benchmark_hi.jsonl",
        help="Path to Hindi benchmark JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmark_results",
        help="Directory to save experiment outputs and audit report",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between API calls",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Only run preflight checks and print resource estimates without generating",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume; rerun all conditions from scratch",
    )

    args = parser.parse_args()
    en_p = Path(args.en_benchmark)
    hi_p = Path(args.hi_benchmark)

    print("=" * 70)
    print("FactIndic Six-Condition Benchmark Experiment Pipeline")
    print("=" * 70)

    preflight = preflight_check(en_p, hi_p)
    print(f"[*] Preflight Check:")
    print(f"    - Ready to Run : {preflight['ready']}")
    print(f"    - English Count: {preflight['en_count']}")
    print(f"    - Hindi Count  : {preflight['hi_count']}")
    print(f"    - Total Records: {preflight['total_source_records']}")
    print(f"    - Total Evals  : {preflight['total_condition_evaluations']} (6 conditions per record)")
    print(f"    - Est. API Calls: ~{preflight['estimated_total_api_calls']} calls")
    print(f"    - Model ID     : {preflight['model_id']}")

    if not preflight["ready"]:
        print("\n[X] PREFLIGHT BLOCKERS DETECTED:")
        for b in preflight["blockers"]:
            print(f"    * {b}")
        sys.exit(1)

    if args.preflight_only:
        print("\n[✓] Preflight check completed successfully (--preflight-only specified). Exiting.")
        sys.exit(0)

    try:
        execute_benchmark_experiment(
            en_benchmark_path=en_p,
            hi_benchmark_path=hi_p,
            output_dir=args.output_dir,
            delay_seconds=args.delay,
            resume=not args.no_resume,
        )
        print("\n[✓] Benchmark experiment completed successfully.")
    except Exception as exc:
        print(f"\n[X] Benchmark experiment failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
