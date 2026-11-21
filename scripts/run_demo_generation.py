#!/usr/bin/env python3
"""FactIndic Phase 2.1: Demo-Only Summary Generation Experiment.

Strict Research Policy:
- This is a TEMPORARY DEMO RUN labeled 'demo_only'.
- The final research model decision remains explicitly 'UNDECIDED'.
- Evaluates Prompt A vs Prompt B on genuine XL-Sum test records.
- Does not alter the final research scope or benchmark comparisons.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from factindic.datasets.xlsum import XLSumLoader
from factindic.generation import GenerationRecord, run_article_generation, save_generation_records
from factindic.models.alternative import AlternativeDemoClient
from factindic.prompts.templates import PROMPT_TEMPLATE_VERSION, PromptStrategy
from factindic.schema import FactIndicRecord

TARGET_RECORD_IDS = {
    "english": ["world-us-canada-38697908", "uk-wales-politics-48586353"],
    "hindi": ["081027_sadhvi_test_pp", "070929_pakistan_protest"],
}


def retrieve_target_xlsum_records(language: str, split: str = "test") -> List[FactIndicRecord]:
    """Retrieve the exact target records from live Hugging Face csebuetnlp/xlsum."""
    loader = XLSumLoader(language=language)
    target_ids = set(TARGET_RECORD_IDS.get(loader.hf_config, []))
    matched_records = []

    print(f"[*] Ingesting live records from Hugging Face 'csebuetnlp/xlsum' ({loader.hf_config})...")
    print(f"[*] Target record IDs: {target_ids}")

    # Use stream_raw to inspect genuine IDs
    for raw_item, record in loader.stream_raw(split=split, max_samples=20):
        raw_id = raw_item.get("id")
        if raw_id in target_ids:
            matched_records.append(record)
            print(f"    [✓] Matched Target Record: ID='{raw_id}' Title='{record.metadata.get('title', '')[:60]}'")
            if len(matched_records) >= len(target_ids):
                break

    return matched_records


def run_demo_generation(
    client: Optional[AlternativeDemoClient],
    client_error: Optional[str],
    provider_name: str,
    model_name: str,
    output_dir: Path,
    temperature: float = 0.3,
    max_new_tokens: int = 256,
    top_p: float = 0.9,
) -> Dict[str, Any]:
    """Execute Prompt A and Prompt B across target English and Hindi articles."""
    results_jsonl = output_dir / "factindic_demo_only_results.jsonl"
    audit_json = output_dir / "factindic_demo_only_audit.json"

    audit: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "demo_only",
        "final_research_model": "UNDECIDED",
        "demo_model_identifier": model_name,
        "demo_provider": provider_name,
        "generation_parameters": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
        },
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "dataset_repository": "csebuetnlp/xlsum",
        "languages": {},
        "summary_counts": {
            "total_articles": 0,
            "prompt_a_success": 0,
            "prompt_a_failed": 0,
            "prompt_b_success": 0,
            "prompt_b_failed": 0,
        },
        "client_error": client_error,
        "results_file": str(results_jsonl),
        "audit_file": str(audit_json),
    }

    all_generated_records: List[GenerationRecord] = []

    for lang in ["english", "hindi"]:
        lang_audit: Dict[str, Any] = {
            "articles_retrieved": 0,
            "retrieved_ids": [],
            "missing_ids": [],
            "prompt_a_success": 0,
            "prompt_a_failed": 0,
            "prompt_b_success": 0,
            "prompt_b_failed": 0,
        }

        # 1. Retrieve genuine records
        try:
            articles = retrieve_target_xlsum_records(language=lang, split="test")
        except Exception as exc:
            lang_audit["error"] = f"Retrieval failed: {exc}"
            audit["languages"][lang] = lang_audit
            continue

        retrieved_ids = [a.metadata.get("raw_id") for a in articles]
        target_ids = TARGET_RECORD_IDS.get(lang, [])
        missing = [tid for tid in target_ids if tid not in retrieved_ids]

        lang_audit["articles_retrieved"] = len(articles)
        lang_audit["retrieved_ids"] = retrieved_ids
        lang_audit["missing_ids"] = missing
        audit["summary_counts"]["total_articles"] += len(articles)

        # 2. Check client availability
        if client is None:
            lang_audit["error"] = f"Inference blocked: {client_error}"
            lang_audit["prompt_a_failed"] = len(articles)
            lang_audit["prompt_b_failed"] = len(articles)
            audit["summary_counts"]["prompt_a_failed"] += len(articles)
            audit["summary_counts"]["prompt_b_failed"] += len(articles)
            audit["languages"][lang] = lang_audit
            continue

        # 3. Generate summaries under Prompt A and Prompt B
        for idx, art in enumerate(articles, 1):
            art_id = art.metadata.get("raw_id", art.example_id)
            print(f"\n[*] Processing Article #{idx} (Language: '{lang}', ID: {art_id})...")

            # Condition 1: Prompt A
            print("    -> Generating Prompt A (Standard Summarization)...")
            rec_a = run_article_generation(
                record=art,
                client=client,
                strategy=PromptStrategy.PROMPT_A,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
            )
            if rec_a.status == "SUCCESS":
                lang_audit["prompt_a_success"] += 1
                audit["summary_counts"]["prompt_a_success"] += 1
                print(f"       [✓] Generated Prompt A summary ({len(rec_a.generated_summary)} chars, {rec_a.latency_seconds}s)")
            else:
                lang_audit["prompt_a_failed"] += 1
                audit["summary_counts"]["prompt_a_failed"] += 1
                print(f"       [X] Prompt A generation failed: {rec_a.error_message}")
            all_generated_records.append(rec_a)

            # Condition 2: Prompt B
            print("    -> Generating Prompt B (Factuality-Aware Summarization)...")
            rec_b = run_article_generation(
                record=art,
                client=client,
                strategy=PromptStrategy.PROMPT_B,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
            )
            if rec_b.status == "SUCCESS":
                lang_audit["prompt_b_success"] += 1
                audit["summary_counts"]["prompt_b_success"] += 1
                print(f"       [✓] Generated Prompt B summary ({len(rec_b.generated_summary)} chars, {rec_b.latency_seconds}s)")
            else:
                lang_audit["prompt_b_failed"] += 1
                audit["summary_counts"]["prompt_b_failed"] += 1
                print(f"       [X] Prompt B generation failed: {rec_b.error_message}")
            all_generated_records.append(rec_b)

        audit["languages"][lang] = lang_audit

    # Save outputs if any successful records exist
    successful_records = [r for r in all_generated_records if r.status == "SUCCESS"]
    if successful_records:
        save_generation_records(successful_records, results_jsonl, append=False)
        print(f"\n[*] Saved {len(successful_records)} successful records to: {results_jsonl}")

    # Always save audit log
    with open(audit_json, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    print(f"[*] Saved demo audit report to: {audit_json}\n")

    return audit


def main():
    parser = argparse.ArgumentParser(description="FactIndic Demo-Only Summary Generation")
    parser.add_argument("--provider", type=str, default=None, help="LLM Provider ('google', 'groq', 'openai', etc.)")
    parser.add_argument("--model", type=str, default=None, help="Explicit model identifier")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--output-dir", type=str, default=str(ROOT_DIR / "data" / "generations"))

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = None
    client_error = None
    provider_name = "NONE"
    model_name = "NONE"

    try:
        client = AlternativeDemoClient(provider=args.provider, model_id=args.model)
        provider_name = client.provider
        model_name = client.model_id
        print(f"[*] Initialized AlternativeDemoClient for demo_only run (provider='{provider_name}', model='{model_name}')")
    except Exception as exc:
        client_error = str(exc)
        print(f"[!] Info: AlternativeDemoClient could not be initialized.\n    Reason: {client_error}\n")

    audit = run_demo_generation(
        client=client,
        client_error=client_error,
        provider_name=provider_name,
        model_name=model_name,
        output_dir=output_dir,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
    )

    print(f"{'='*75}")
    print(f"FactIndic Demo-Only Summary Generation Run Completed.")
    print(f"Final Research Model Setting : {audit['final_research_model']}")
    print(f"Demo Model Used              : {audit['demo_model_identifier']} ({audit['demo_provider']})")
    print(f"Total Source Articles        : {audit['summary_counts']['total_articles']}")
    print(f"Prompt A Summaries Generated : {audit['summary_counts']['prompt_a_success']}")
    print(f"Prompt B Summaries Generated : {audit['summary_counts']['prompt_b_success']}")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
