#!/usr/bin/env python3
"""Preflight verification command-line interface for FactIndic benchmark experiments.

Usage:
    python scripts/preflight_check.py [--manifest PATH] [--records N] [--json]

Examples:
    # Full 200-record benchmark preflight
    python scripts/preflight_check.py

    # Controlled 10-record subset preflight
    python scripts/preflight_check.py --records 10
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factindic.preflight import run_preflight


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run preflight checks and quota budget estimation for FactIndic experiments."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/benchmark/benchmark_manifest.json",
        help="Path to benchmark_manifest.json (default: data/benchmark/benchmark_manifest.json)",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default=None,
        help="Directory containing benchmark files (default: parent of manifest)",
    )
    parser.add_argument(
        "--records",
        type=int,
        default=None,
        help="Target number of records (default: total in manifest, or 200)",
    )
    parser.add_argument(
        "--allow-insufficient-quota",
        action="store_true",
        help="Do not block execution when estimated calls exceed trial limit (warning only)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results strictly as machine-readable JSON",
    )

    args = parser.parse_args()

    report = run_preflight(
        manifest_path=args.manifest,
        benchmark_dir=args.benchmark_dir,
        target_records=args.records,
        allow_insufficient_quota=args.allow_insufficient_quota,
    )

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["can_proceed"] else 1

    # Human-readable format
    print("\n" + "=" * 70)
    print("           FACTINDIC BENCHMARK PREFLIGHT REPORT")
    print("=" * 70)

    # Status banner
    status = report["status"]
    if status == "READY":
        print(f"Status: \033[92mREADY TO RUN\033[0m")
    else:
        print(f"Status: \033[91mBLOCKED ({len(report['blockers'])} issue(s) detected)\033[0m")

    # Manifest section
    m_info = report["manifest"]
    print("\n1. Benchmark Manifest & Data Files:")
    print(f"   Manifest: {m_info['path']}")
    print(f"   Integrity: {'VALID (Checksums & IDs verified)' if m_info['is_valid'] else 'INVALID'}")
    if m_info["errors"]:
        for err in m_info["errors"]:
            print(f"     - [FAIL] {err}")

    # Credentials
    c_info = report["credentials"]
    print("\n2. API Credentials & Configuration:")
    print(f"   Cohere Key: {'Configured' if c_info['configured'] else 'NOT Configured'}")
    print(f"   Key Length: {c_info['key_length']} characters")

    # Budget Breakdown
    b = report["budget"]
    print(f"\n3. API Call Budget ({b['num_records']} records, {b['prompts_per_record']} prompts/record):")
    print(f"   - Generation calls:        {b['generation_calls']:,}")
    print(f"   - Whole-document NLI:      {b['whole_doc_nli_calls']:,}")
    print(f"   - Sentence-level NLI:      {b['sentence_nli_calls']:,} (~{b['avg_sentences_per_summary']} sents/summary)")
    print(f"   - Claim extraction calls:  {b['claim_extraction_calls']:,}")
    print(f"   - Hybrid claim NLI calls:  {b['hybrid_claim_nli_calls']:,} (~{b['avg_claims_per_sentence']} claims/sent x {b['top_k_evidence']} evidence)")
    print(f"   ----------------------------------------")
    print(f"   TOTAL ESTIMATED CALLS:     {b['total_calls']:,}")
    print(f"   Estimated Runtime (Trial, 20 RPM):      ~{b['estimated_minutes_trial']} mins ({b['estimated_hours_trial']} hrs)")
    print(f"   Estimated Runtime (Production, 100 RPM): ~{b['estimated_minutes_production']} mins")

    # Storage
    s_info = report["storage"]
    print(f"\n4. Storage & System:")
    print(f"   Available Disk Space: {s_info['free_disk_gb']} GB")

    # Blockers / Next steps
    if report["blockers"]:
        print("\nBlockers to resolve before execution:")
        for idx, blocker in enumerate(report["blockers"], start=1):
            print(f"   {idx}. {blocker}")
        print("\nPreflight: FAILED")
        return 1
    else:
        print("\nAll preflight checks PASSED. Environment is ready for execution.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
