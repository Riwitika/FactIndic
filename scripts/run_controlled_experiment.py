#!/usr/bin/env python3
"""Run the FactIndic Controlled Summary Generation Experiment (Phase 1).

Executes Prompt A and Prompt B across the 10 genuine XL-Sum test records
(5 English, 5 Hindi) using Cohere Command A+ (command-a-plus-05-2026).
"""

import argparse
import sys
from pathlib import Path

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from factindic.controlled_experiment import execute_controlled_experiment
from factindic.models.aya import AyaExpanseClient
from factindic.serialization import load_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="FactIndic Controlled Summary Generation Experiment (Phase 1)")
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between successive API calls (default: 1.0)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Maximum retries per failed API call (default: 1)",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=str(ROOT_DIR / "data" / "generations" / "controlled_generation_results.jsonl"),
        help="Destination path for results JSONL",
    )
    parser.add_argument(
        "--audit-file",
        type=str,
        default=str(ROOT_DIR / "data" / "generations" / "controlled_generation_audit.json"),
        help="Destination path for audit JSON report",
    )

    args = parser.parse_args()

    # 1. Initialize client
    print("=" * 75)
    print("FactIndic Controlled Summary Generation Experiment (Phase 1)")
    print("=" * 75)
    print("[*] Initializing generation client...")
    try:
        client = AyaExpanseClient()
        print(f"[✓] Client initialized: provider='{client.provider}', model='{client.model_id}'")
        print(f"    Credentials: {'[SECURELY LOADED]' if client.api_key else '[NONE]'}")
    except Exception as exc:
        print(f"[X] FATAL: Client initialization failed: {exc}", file=sys.stderr)
        return 1

    # 2. Load genuine source records
    en_path = ROOT_DIR / "data" / "samples" / "factindic_en_test_sample_5.jsonl"
    hi_path = ROOT_DIR / "data" / "samples" / "factindic_hi_test_sample_5.jsonl"

    if not en_path.exists() or not hi_path.exists():
        print(f"[X] FATAL: Sample files missing in {ROOT_DIR / 'data' / 'samples'}", file=sys.stderr)
        return 1

    en_records = load_jsonl(en_path)
    hi_records = load_jsonl(hi_path)

    print(f"[*] Loaded {len(en_records)} English and {len(hi_records)} Hindi genuine XL-Sum records.")
    print(f"    Output results: {args.results_file}")
    print(f"    Output audit  : {args.audit_file}")
    print("-" * 75)

    # 3. Execute experiment
    records, audit = execute_controlled_experiment(
        client=client,
        en_records=en_records,
        hi_records=hi_records,
        results_path=args.results_file,
        audit_path=args.audit_file,
        delay_seconds=args.delay,
        max_retries=args.retries,
    )

    print("-" * 75)
    print(f"Experiment Status Summary:")
    print(f"  Total Expected Summaries : {audit['expected_counts']['total_summaries']}")
    print(f"  Total Generated Records  : {len(records)}")
    print(f"  Validation Status        : {'PASSED' if audit['validation_passed'] else 'FAILED'}")
    print(f"  Successful Generations   : {audit['actual_counts']['total_success']}")
    print(f"  Failed Generations       : {audit['actual_counts']['total_failed']}")
    print(f"  English Prompt A Success : {audit['actual_counts']['english']['prompt_a_success']} / 5")
    print(f"  English Prompt B Success : {audit['actual_counts']['english']['prompt_b_success']} / 5")
    print(f"  Hindi Prompt A Success   : {audit['actual_counts']['hindi']['prompt_a_success']} / 5")
    print(f"  Hindi Prompt B Success   : {audit['actual_counts']['hindi']['prompt_b_success']} / 5")
    print("=" * 75)

    if not audit["validation_passed"]:
        print("[!] Validation errors detected:", file=sys.stderr)
        for err in audit["validation_errors"]:
            print(f"    - {err}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
