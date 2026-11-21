"""Controlled summary generation experiment logic, validation, and provenance tracking.

Implements Phase 1 controlled generation:
- 10 genuine XL-Sum source records (5 English, 5 Hindi)
- 2 conditions: Prompt A (standard) and Prompt B (factuality-aware)
- 20 total summaries generated with Cohere Command A+ (command-a-plus-05-2026)
- Resumable without duplication, strict error recording, and zero credential logging
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from factindic.generation import (
    GenerationRecord,
    load_generation_records,
    run_article_generation,
    save_generation_records,
)
from factindic.models.base import BaseLLMClient
from factindic.prompts.templates import PROMPT_TEMPLATE_VERSION, PromptStrategy
from factindic.schema import FactIndicRecord

CONTROLLED_MODEL_ID = "command-a-plus-05-2026"
CONTROLLED_PROVIDER = "cohere"
CONTROLLED_GENERATION_PARAMS: Dict[str, Any] = {
    "temperature": 0.3,
    "max_new_tokens": 256,
    "top_p": 0.9,
}
TOTAL_EXPECTED_SUMMARIES = 20
EXPECTED_PER_LANGUAGE = 10
EXPECTED_PER_PROMPT = 10


def validate_controlled_records(
    records: List[GenerationRecord],
    expected_record_ids: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Validate completeness, uniqueness, and correctness of controlled generation records.

    Args:
        records: List of GenerationRecord instances.
        expected_record_ids: Dict mapping 'english' and 'hindi' to lists of expected raw IDs.

    Returns:
        Structured dictionary containing validation status, errors, and counts breakdown.
    """
    errors: List[str] = []

    # 1. Total count check
    if len(records) != TOTAL_EXPECTED_SUMMARIES:
        errors.append(f"Expected {TOTAL_EXPECTED_SUMMARIES} records, found {len(records)}.")

    # 2. Duplicate detection
    seen_pairs = set()
    duplicates = []
    for r in records:
        pair = (r.source_record_id, r.prompt_strategy)
        if pair in seen_pairs:
            duplicates.append(pair)
        seen_pairs.add(pair)
    if duplicates:
        errors.append(f"Duplicate (source_record_id, prompt_strategy) pairs detected: {duplicates}")

    # 3. Completeness across languages and strategies
    record_map: Dict[Tuple[str, str], GenerationRecord] = {
        (r.source_record_id, r.prompt_strategy): r for r in records
    }

    counts: Dict[str, Any] = {
        "english": {"prompt_a_success": 0, "prompt_a_failed": 0, "prompt_b_success": 0, "prompt_b_failed": 0},
        "hindi": {"prompt_a_success": 0, "prompt_a_failed": 0, "prompt_b_success": 0, "prompt_b_failed": 0},
        "prompt_a_total_success": 0,
        "prompt_b_total_success": 0,
        "total_success": 0,
        "total_failed": 0,
    }

    for lang_key, expected_ids in expected_record_ids.items():
        lang_str = "english" if lang_key.lower() in ("english", "en") else "hindi"
        for rid in expected_ids:
            for strat in ("prompt_a", "prompt_b"):
                key = (rid, strat)
                if key not in record_map:
                    errors.append(f"Missing generation for record ID '{rid}' under {strat} ({lang_str}).")
                    continue

                rec = record_map[key]
                strat_suffix = "a" if strat == "prompt_a" else "b"

                # Check model ID and provider
                if rec.model_id != CONTROLLED_MODEL_ID:
                    errors.append(f"Record {key} has model_id '{rec.model_id}' (expected '{CONTROLLED_MODEL_ID}').")
                if rec.provider != CONTROLLED_PROVIDER:
                    errors.append(f"Record {key} has provider '{rec.provider}' (expected '{CONTROLLED_PROVIDER}').")

                # Check generation parameters
                if rec.generation_parameters != CONTROLLED_GENERATION_PARAMS:
                    errors.append(f"Record {key} has non-standard parameters: {rec.generation_parameters}")

                # Check status and non-empty output
                if rec.status == "SUCCESS":
                    if not rec.generated_summary or not rec.generated_summary.strip():
                        errors.append(f"Record {key} marked SUCCESS but contains empty generated_summary.")
                        counts[lang_str][f"prompt_{strat_suffix}_failed"] += 1
                        counts["total_failed"] += 1
                    else:
                        counts[lang_str][f"prompt_{strat_suffix}_success"] += 1
                        counts[f"prompt_{strat_suffix}_total_success"] += 1
                        counts["total_success"] += 1
                else:
                    errors.append(f"Record {key} failed with error: {rec.error_message}")
                    counts[lang_str][f"prompt_{strat_suffix}_failed"] += 1
                    counts["total_failed"] += 1

    is_valid = len(errors) == 0 and counts["total_success"] == TOTAL_EXPECTED_SUMMARIES

    return {
        "is_valid": is_valid,
        "total_records": len(records),
        "expected_records": TOTAL_EXPECTED_SUMMARIES,
        "counts": counts,
        "errors": errors,
    }


def execute_controlled_experiment(
    client: BaseLLMClient,
    en_records: List[FactIndicRecord],
    hi_records: List[FactIndicRecord],
    results_path: Union[str, Path],
    audit_path: Union[str, Path],
    delay_seconds: float = 1.0,
    max_retries: int = 1,
) -> Tuple[List[GenerationRecord], Dict[str, Any]]:
    """Execute the controlled generation experiment across 10 records under Prompt A and B.

    Resumable: skips already successfully generated (source_record_id, prompt_strategy) pairs.

    Args:
        client: Configured LLM client (AyaExpanseClient / CohereCommandClient).
        en_records: 5 genuine English FactIndicRecord instances.
        hi_records: 5 genuine Hindi FactIndicRecord instances.
        results_path: File path to save output JSONL records.
        audit_path: File path to save audit JSON report.
        delay_seconds: Polite delay between API calls in seconds.
        max_retries: Maximum retries per failed call before recording failure.

    Returns:
        Tuple of (list of all 20 GenerationRecord objects, parsed audit dictionary).
    """
    res_path = Path(results_path)
    aud_path = Path(audit_path)
    res_path.parent.mkdir(parents=True, exist_ok=True)
    aud_path.parent.mkdir(parents=True, exist_ok=True)

    start_time_utc = datetime.now(timezone.utc).isoformat()
    t_start = time.time()

    # 1. Load existing records for resumption
    existing_records: List[GenerationRecord] = []
    if res_path.exists():
        try:
            existing_records = load_generation_records(res_path)
        except Exception:
            existing_records = []

    # Map existing successful records
    record_store: Dict[Tuple[str, str], GenerationRecord] = {}
    for r in existing_records:
        if r.status == "SUCCESS" and r.generated_summary and r.generated_summary.strip():
            record_store[(r.source_record_id, r.prompt_strategy)] = r

    expected_ids = {
        "english": [r.metadata.get("raw_id", r.example_id) for r in en_records],
        "hindi": [r.metadata.get("raw_id", r.example_id) for r in hi_records],
    }

    datasets_to_process = [
        ("english", en_records),
        ("hindi", hi_records),
    ]

    total_runs = len(en_records) * 2 + len(hi_records) * 2
    processed_count = 0

    print(f"[*] Starting Controlled Generation Experiment ({total_runs} total target conditions)")
    print(f"    Resumed with {len(record_store)} already-completed conditions.")

    for lang, records in datasets_to_process:
        for article in records:
            raw_id = article.metadata.get("raw_id", article.example_id)

            for strategy in (PromptStrategy.PROMPT_A, PromptStrategy.PROMPT_B):
                strat_key = strategy.value
                key = (raw_id, strat_key)
                processed_count += 1

                # Check if already completed
                if key in record_store:
                    existing = record_store[key]
                    print(f"    [{processed_count}/{total_runs}] SKIP (Already Completed): {lang.upper()} | ID='{raw_id}' | {strat_key}")
                    continue

                print(f"    [{processed_count}/{total_runs}] GENERATING: {lang.upper()} | ID='{raw_id}' | {strat_key}...")

                # Execute with bounded retries
                gen_rec: Optional[GenerationRecord] = None
                attempts = 0
                while attempts <= max_retries:
                    attempts += 1
                    gen_rec = run_article_generation(
                        record=article,
                        client=client,
                        strategy=strategy,
                        temperature=CONTROLLED_GENERATION_PARAMS["temperature"],
                        max_new_tokens=CONTROLLED_GENERATION_PARAMS["max_new_tokens"],
                        top_p=CONTROLLED_GENERATION_PARAMS["top_p"],
                    )
                    if gen_rec.status == "SUCCESS" and gen_rec.generated_summary.strip():
                        break
                    if attempts <= max_retries:
                        print(f"        [!] Transient failure on attempt {attempts}, retrying after 2s...")
                        time.sleep(2.0)

                assert gen_rec is not None
                record_store[key] = gen_rec

                if gen_rec.status == "SUCCESS":
                    print(f"        [✓] SUCCESS ({gen_rec.latency_seconds}s, {len(gen_rec.generated_summary)} chars)")
                else:
                    print(f"        [X] FAILED: {gen_rec.error_message}")

                # Save incrementally to protect progress
                all_current = list(record_store.values())
                save_generation_records(all_current, res_path, append=False)

                # Polite rate-limiting delay
                if delay_seconds > 0:
                    time.sleep(delay_seconds)

    # 2. Assemble ordered records
    final_records: List[GenerationRecord] = []
    for lang, records in datasets_to_process:
        for article in records:
            raw_id = article.metadata.get("raw_id", article.example_id)
            for strat in (PromptStrategy.PROMPT_A.value, PromptStrategy.PROMPT_B.value):
                key = (raw_id, strat)
                if key in record_store:
                    final_records.append(record_store[key])

    # Save complete finalized file
    save_generation_records(final_records, res_path, append=False)

    # 3. Validate results
    validation_report = validate_controlled_records(final_records, expected_ids)

    end_time_utc = datetime.now(timezone.utc).isoformat()
    duration_seconds = round(time.time() - t_start, 3)

    # 4. Construct audit log
    audit_data = {
        "timestamp_start_utc": start_time_utc,
        "timestamp_end_utc": end_time_utc,
        "total_duration_seconds": duration_seconds,
        "experiment_name": "FactIndic Controlled Summary Generation Experiment (Phase 1)",
        "model_configuration": {
            "provider": CONTROLLED_PROVIDER,
            "model_id": CONTROLLED_MODEL_ID,
            "generation_parameters": CONTROLLED_GENERATION_PARAMS,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        },
        "dataset_provenance": {
            "repository": "csebuetnlp/xlsum",
            "split": "test",
            "english_sample_size": len(en_records),
            "hindi_sample_size": len(hi_records),
            "total_source_articles": len(en_records) + len(hi_records),
            "english_record_ids": expected_ids["english"],
            "hindi_record_ids": expected_ids["hindi"],
        },
        "expected_counts": {
            "total_summaries": TOTAL_EXPECTED_SUMMARIES,
            "english_total": EXPECTED_PER_LANGUAGE,
            "hindi_total": EXPECTED_PER_LANGUAGE,
            "prompt_a_total": EXPECTED_PER_PROMPT,
            "prompt_b_total": EXPECTED_PER_PROMPT,
            "english_prompt_a": len(en_records),
            "english_prompt_b": len(en_records),
            "hindi_prompt_a": len(hi_records),
            "hindi_prompt_b": len(hi_records),
        },
        "actual_counts": validation_report["counts"],
        "validation_passed": validation_report["is_valid"],
        "validation_errors": validation_report["errors"],
        "results_file": str(res_path),
        "audit_file": str(aud_path),
    }

    with open(aud_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)

    print(f"[*] Controlled generation experiment completed in {duration_seconds}s.")
    print(f"    Validation Passed: {validation_report['is_valid']}")
    print(f"    Records Written  : {len(final_records)}")
    print(f"    Results Path     : {res_path}")
    print(f"    Audit Path       : {aud_path}")

    return final_records, audit_data
