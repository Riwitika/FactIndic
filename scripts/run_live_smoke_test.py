#!/usr/bin/env python3
"""FactIndic Controlled Live Inference Smoke Test for Cohere Command A+.

Strict Policies:
- Loads COHERE_API_KEY securely from environment or .env (never hardcodes or logs keys).
- Uses genuine XL-Sum test records from data/samples/.
- Runs 1 English and 1 Hindi summarization request via standard pipeline.
- Validates non-empty output and saves results in standard GenerationRecord schema.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from factindic.generation import GenerationRecord, run_article_generation, save_generation_records
from factindic.models.aya import AyaExpanseClient
from factindic.prompts.templates import PromptStrategy
from factindic.serialization import load_jsonl


def main() -> int:
    print("=" * 75)
    print("FactIndic Controlled Live Smoke Test (Cohere Command A+)")
    print("=" * 75)

    output_dir = ROOT_DIR / "data" / "generations"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "live_smoke_test_results.jsonl"
    audit_path = output_dir / "live_smoke_test_audit.json"

    # 1. Initialize client securely
    print("[*] Initializing generation client...")
    try:
        client = AyaExpanseClient()
        print(f"[✓] Client initialized: provider='{client.provider}', model='{client.model_id}'")
        print(f"    API Key detected: {'[YES - SECURELY LOADED]' if client.api_key else '[NO]'}")
    except Exception as exc:
        print(f"[X] FATAL: Client initialization failed: {exc}", file=sys.stderr)
        return 1

    # 2. Load genuine XL-Sum samples
    en_sample_file = ROOT_DIR / "data" / "samples" / "factindic_en_test_sample_5.jsonl"
    hi_sample_file = ROOT_DIR / "data" / "samples" / "factindic_hi_test_sample_5.jsonl"

    if not en_sample_file.exists() or not hi_sample_file.exists():
        print(f"[X] FATAL: Sample files missing in {ROOT_DIR / 'data' / 'samples'}", file=sys.stderr)
        return 1

    en_records = load_jsonl(en_sample_file)
    hi_records = load_jsonl(hi_sample_file)

    target_en = en_records[0]
    target_hi = hi_records[0]

    tasks = [
        ("english", target_en),
        ("hindi", target_hi),
    ]

    generated_records = []
    audit_results = {}
    all_succeeded = True

    # 3. Execute live generation
    for lang, record in tasks:
        raw_id = record.metadata.get("raw_id", record.example_id)
        print(f"\n[*] Sending live summarization request for {lang.upper()} (ID: {raw_id})...")
        print(f"    Source text length: {len(record.source_text)} characters")

        t0 = time.time()
        rec = run_article_generation(
            record=record,
            client=client,
            strategy=PromptStrategy.PROMPT_A,
            temperature=0.3,
            max_new_tokens=256,
            top_p=0.9,
        )
        duration = round(time.time() - t0, 3)

        success = rec.status == "SUCCESS" and bool(rec.generated_summary and rec.generated_summary.strip())
        if not success:
            all_succeeded = False

        audit_results[lang] = {
            "record_id": raw_id,
            "example_id": record.example_id,
            "status": rec.status,
            "is_non_empty": bool(rec.generated_summary and rec.generated_summary.strip()),
            "summary_char_length": len(rec.generated_summary) if rec.generated_summary else 0,
            "latency_seconds": rec.latency_seconds,
            "error_message": rec.error_message,
        }

        if success:
            print(f"[✓] {lang.upper()} SUCCESS ({rec.latency_seconds}s)")
            print(f"    Summary ({len(rec.generated_summary)} chars): {rec.generated_summary[:120]}...")
        else:
            print(f"[X] {lang.upper()} FAILED: {rec.error_message}", file=sys.stderr)

        generated_records.append(rec)

    # 4. Save results to JSONL
    save_generation_records(generated_records, results_path, append=False)
    print(f"\n[*] Saved {len(generated_records)} records to: {results_path}")

    # 5. Save audit metadata (zero credentials logged)
    audit_data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": client.model_id,
        "provider": client.provider,
        "generation_parameters": {
            "temperature": 0.3,
            "max_new_tokens": 256,
            "top_p": 0.9,
        },
        "all_succeeded": all_succeeded,
        "results": audit_results,
        "results_file": str(results_path),
    }

    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)
    print(f"[*] Saved audit metadata to: {audit_path}")
    print("=" * 75)

    return 0 if all_succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
