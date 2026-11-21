#!/usr/bin/env python3
"""Run Phase 2 real-data summary generation experiment for FactIndic.

Evaluates Prompt A (Standard) vs Prompt B (Factuality-Aware) using genuine
records from Hugging Face csebuetnlp/xlsum (English and Hindi).

Strict Policy:
- Connects only to live Hugging Face XL-Sum.
- Never falls back to mock fixtures or synthetic data.
- Keeps generation parameters identical between Prompt A and Prompt B.
- Records full provenance and errors.
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
from factindic.generation import GenerationRecord, run_article_generation, save_generation_records
from factindic.models.aya import AyaExpanseClient
from factindic.prompts.templates import PromptStrategy


def run_experiment_for_language(
    language: str,
    split: str,
    sample_size: int,
    client: Any,
    client_init_error: str,
    output_path: Path,
    temperature: float = 0.3,
    max_new_tokens: int = 256,
    top_p: float = 0.9,
) -> Dict[str, Any]:
    """Execute Prompt A and Prompt B generation for a specific language."""
    print(f"\n{'='*75}")
    print(f"Summary Generation Experiment: Language='{language}', Split='{split}'")
    print(f"{'='*75}")

    lang_audit: Dict[str, Any] = {
        "language": language,
        "split": split,
        "sample_size_requested": sample_size,
        "articles_retrieved_count": 0,
        "retrieved_record_ids": [],
        "prompt_a_success": 0,
        "prompt_a_failed": 0,
        "prompt_b_success": 0,
        "prompt_b_failed": 0,
        "error": None,
    }

    # 1. Fetch genuine live records from Hugging Face
    print(f"[*] Ingesting {sample_size} genuine live records from 'csebuetnlp/xlsum' ({language})...")
    loader = XLSumLoader(language=language)
    try:
        articles = loader.load(split=split, max_samples=sample_size)
    except Exception as exc:
        print(f"[X] FATAL: Failed to retrieve live XL-Sum records for '{language}': {exc}", file=sys.stderr)
        lang_audit["error"] = f"Live XL-Sum retrieval failure: {exc}"
        return lang_audit

    lang_audit["articles_retrieved_count"] = len(articles)
    lang_audit["retrieved_record_ids"] = [a.metadata.get("raw_id", a.example_id) for a in articles]
    print(f"[✓] Retrieved {len(articles)} genuine articles from Hugging Face.")
    for idx, art in enumerate(articles, 1):
        print(f"    Article #{idx}: ID='{art.metadata.get('raw_id')}' Title='{art.metadata.get('title')[:60]}'")

    # 2. Check if inference client is available
    if client is None:
        print(f"\n[!] INFERENCE BLOCKED for '{language}':")
        print(f"    {client_init_error}")
        lang_audit["error"] = f"Inference client unavailable: {client_init_error}"
        lang_audit["prompt_a_failed"] = len(articles)
        lang_audit["prompt_b_failed"] = len(articles)
        return lang_audit

    # 3. Execute generation under Prompt A and Prompt B
    generated_records: List[GenerationRecord] = []

    for idx, art in enumerate(articles, 1):
        art_id = art.metadata.get("raw_id", art.example_id)
        print(f"\n[*] Processing Article #{idx} (ID: {art_id})...")

        # Condition 1: Prompt A (Standard)
        print("    -> Running Condition 1: Prompt A (Standard Summarization)...")
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
            print(f"       [✓] Generated summary ({len(rec_a.generated_summary)} chars, latency: {rec_a.latency_seconds}s)")
        else:
            lang_audit["prompt_a_failed"] += 1
            print(f"       [X] Prompt A failed: {rec_a.error_message}")
        generated_records.append(rec_a)

        # Condition 2: Prompt B (Factuality-Aware)
        print("    -> Running Condition 2: Prompt B (Factuality-Aware Summarization)...")
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
            print(f"       [✓] Generated summary ({len(rec_b.generated_summary)} chars, latency: {rec_b.latency_seconds}s)")
        else:
            lang_audit["prompt_b_failed"] += 1
            print(f"       [X] Prompt B failed: {rec_b.error_message}")
        generated_records.append(rec_b)

    # 4. Save results
    if generated_records:
        written = save_generation_records(generated_records, output_path, append=True)
        print(f"\n[*] Saved {written} generation records to: {output_path}")

    return lang_audit


def main():
    parser = argparse.ArgumentParser(description="FactIndic Summary Generation Experiment (Phase 2)")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["english", "hindi"],
        help="Languages to process (default: 'english' 'hindi')",
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
        help="Number of articles to process per language (default: 2)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generation length in tokens (default: 256)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top_p (default: 0.9)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Explicit LLM provider ('cohere', 'huggingface', 'openai_compatible')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="command-a-plus-05-2026",
        help="Model ID to evaluate (default: 'command-a-plus-05-2026')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "data" / "generations"),
        help="Output directory for generated results and audit files",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "factindic_smoke_experiment_results.jsonl"
    audit_file = output_dir / "generation_experiment_audit.json"

    # Attempt to initialize generation client
    client = None
    client_init_error = None
    try:
        client = AyaExpanseClient(provider=args.provider, model_id=args.model)
        print(f"[*] Initialized generation client (provider: {client.provider}, model: {client.model_id})")
    except Exception as exc:
        client_init_error = str(exc)
        print(f"[!] Warning: Generation client could not be initialized.\n    Reason: {client_init_error}\n")

    experiment_audit: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "FactIndic Phase 2 Summary Generation Smoke Test",
        "model_requested": f"Command A+ ({args.model})",
        "provider": client.provider if client else "NONE (BLOCKED)",
        "model_id": client.model_id if client else args.model,
        "parameters": {
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "top_p": args.top_p,
        },
        "languages": {},
        "client_init_error": client_init_error,
        "results_file": str(results_file),
    }

    for lang in args.languages:
        audit = run_experiment_for_language(
            language=lang,
            split=args.split,
            sample_size=args.count,
            client=client,
            client_init_error=client_init_error,
            output_path=results_file,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
        )
        experiment_audit["languages"][lang] = audit

    # Write audit log
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(experiment_audit, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*75}")
    print(f"[*] Experiment audit saved to: {audit_file}")
    if results_file.exists():
        print(f"[*] Generated summaries written to: {results_file}")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
