#!/usr/bin/env python3
"""CLI script to re-verify preserved generated summaries without repeating LLM generation.

Re-evaluates C1–C6 conditions across the 10 source records using the existing saved
summaries from data/benchmark_results/controlled_run/experiment_results.jsonl.

Saves results separately under data/benchmark_results/controlled_run/reverified/.
"""

import argparse
import json
import os
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
    CONDITION_SPECS as CONDITION_DEFINITIONS,
    ExperimentRecord,
    load_experiment_records,
    save_experiment_records,
    _get_article_id,
)
from factindic.generation import GenerationRecord, PromptStrategy
from factindic.models.aya import CohereClient
from factindic.schema import FactIndicRecord
from factindic.serialization import load_jsonl
from factindic.verification import (
    AtomicClaimExtractor,
    BM25EvidenceRetriever,
    HybridClaimVerifier,
    SentenceLevelVerifier,
    WholeDocumentVerifier,
    XNLIVerifier,
)
from factindic.verification.claim_extractor import CLAIMS_JSON_SCHEMA


def reverify_controlled_experiment(
    results_path: Union[str, Path] = "data/benchmark_results/controlled_run/experiment_results.jsonl",
    en_benchmark_path: Union[str, Path] = "data/benchmark/controlled_sample_en.jsonl",
    hi_benchmark_path: Union[str, Path] = "data/benchmark/controlled_sample_hi.jsonl",
    output_dir: Union[str, Path] = "data/benchmark_results/controlled_run/reverified",
    delay_seconds: float = 3.0,
    resume: bool = True,
) -> Dict[str, Any]:
    """Re-verify existing generations across all 6 conditions without re-generating summaries.

    Args:
        results_path: Path to existing 60-line experiment results file.
        en_benchmark_path: Path to English source benchmark JSONL.
        hi_benchmark_path: Path to Hindi source benchmark JSONL.
        output_dir: Output directory for reverified results and audit.
        delay_seconds: Delay between claim extraction API calls.
        resume: Whether to resume from partially completed results.

    Returns:
        Structured audit report dictionary.
    """
    res_path = Path(results_path)
    if not res_path.exists():
        raise FileNotFoundError(f"Input experiment results file missing: {res_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_results_file = out_dir / "experiment_results.jsonl"
    out_audit_file = out_dir / "experiment_audit.json"

    # 1. Load benchmark source records for reference
    en_records = load_jsonl(Path(en_benchmark_path))
    hi_records = load_jsonl(Path(hi_benchmark_path))
    source_map: Dict[str, FactIndicRecord] = {}
    for r in en_records + hi_records:
        rec_id = _get_article_id(r)
        source_map[rec_id] = r
        # Also map example_id if different
        if r.example_id:
            source_map[r.example_id] = r

    # 2. Extract preserved generated summaries from existing results
    raw_results = load_experiment_records(res_path)
    preserved_generations: Dict[Tuple[str, str], GenerationRecord] = {}

    for rec in raw_results:
        art_id = rec.source_record_id
        strat_key = rec.prompt_strategy
        key = (art_id, strat_key)

        if key not in preserved_generations and rec.generated_summary:
            src_article = source_map.get(art_id)
            source_text = src_article.source_text if src_article else ""
            ref_summary = src_article.reference_summary if src_article else ""
            src_dataset = src_article.metadata.get("source_dataset", "csebuetnlp/xlsum") if src_article else "csebuetnlp/xlsum"

            gen_rec = GenerationRecord(
                source_dataset=src_dataset,
                source_config="default",
                source_split="test",
                source_record_id=art_id,
                language=rec.language,
                source_text=source_text,
                reference_summary=ref_summary,
                prompt_strategy=strat_key,
                prompt_template_version=rec.prompt_template_version,
                model_id=rec.model_id,
                provider=rec.provider,
                generation_parameters=rec.generation_parameters,
                generated_summary=rec.generated_summary,
                generation_timestamp_utc=rec.timestamp_utc,
                latency_seconds=rec.generation_latency_seconds,
                status="SUCCESS",
            )
            preserved_generations[key] = gen_rec

    print(f"[*] Re-verification Setup:")
    print(f"    - Source Records Identified : {len(source_map)}")
    print(f"    - Preserved Generated Summaries: {len(preserved_generations)} (20 expected: 10 Prompt A, 10 Prompt B)")
    print(f"    - Output Directory          : {out_dir}")

    if len(preserved_generations) != 20:
        print(f"[!] Warning: Expected 20 preserved summaries, found {len(preserved_generations)}")

    # 3. Initialize Verifiers
    llm_client = CohereClient()
    nli_verifier = XNLIVerifier()
    whole_doc_verifier = WholeDocumentVerifier(verifier=nli_verifier)
    sentence_verifier = SentenceLevelVerifier(verifier=nli_verifier)
    claim_extractor = AtomicClaimExtractor(client=llm_client, delay_seconds=delay_seconds)
    evidence_retriever = BM25EvidenceRetriever()
    hybrid_verifier = HybridClaimVerifier(verifier=nli_verifier, retriever=evidence_retriever)

    # 4. Check Resumption
    completed_keys: Dict[Tuple[str, str], ExperimentRecord] = {}
    if resume and out_results_file.exists():
        existing_recs = load_experiment_records(out_results_file)
        for r in existing_recs:
            if r.status == "SUCCESS":
                completed_keys[(r.source_record_id, r.condition_id)] = r
        print(f"[*] Resuming from existing results: {len(completed_keys)} conditions already SUCCESS.")

    all_conditions = [
        ConditionID.C1,
        ConditionID.C2,
        ConditionID.C3,
        ConditionID.C4,
        ConditionID.C5,
        ConditionID.C6,
    ]

    t_start = time.time()
    start_time_utc = datetime.now(timezone.utc).isoformat()
    reverified_results: List[ExperimentRecord] = []

    # Iterate through the 10 source records
    ordered_source_records = en_records + hi_records
    for art_idx, article in enumerate(ordered_source_records, 1):
        art_id = _get_article_id(article)

        for cond_id in all_conditions:
            cond_key = (art_id, cond_id.value)
            if cond_key in completed_keys:
                reverified_results.append(completed_keys[cond_key])
                continue

            cond_def = CONDITION_DEFINITIONS[cond_id]
            strat: PromptStrategy = cond_def["prompt_strategy"]
            verif_strat: str = cond_def["verification_strategy"]
            strat_key = strat.value

            gen_key = (art_id, strat_key)
            if gen_key not in preserved_generations:
                raise RuntimeError(f"Missing preserved summary for {art_id} under {strat_key}")

            gen_rec = preserved_generations[gen_key]

            print(f"[{art_idx}/{len(ordered_source_records)}] Re-verifying {cond_id.value} for {art_id} ({article.language})...", flush=True)

            t_verif_start = time.time()
            verification_results: List[Dict[str, Any]] = []
            status = "SUCCESS"
            error_msg = None

            try:
                if verif_strat == "whole_document":
                    doc_res = whole_doc_verifier.verify_record(gen_rec)
                    verification_results = [doc_res.to_dict()]
                    if doc_res.status == "FAILED":
                        status = "FAILED"
                        error_msg = doc_res.error_message

                elif verif_strat == "sentence_level":
                    sent_results = sentence_verifier.verify_record(gen_rec)
                    verification_results = [r.to_dict() for r in sent_results]
                    if any(r.status == "FAILED" for r in sent_results):
                        if all(r.status == "FAILED" for r in sent_results):
                            status = "FAILED"
                            error_msg = sent_results[0].error_message

                elif verif_strat == "hybrid_claim_level":
                    claims = claim_extractor.extract_from_record(gen_rec)
                    claim_results = hybrid_verifier.verify_record(gen_rec, claims=claims)
                    verification_results = [r.to_dict() for r in claim_results]
                    if any(r.status == "FAILED" for r in claim_results):
                        if all(r.status == "FAILED" for r in claim_results):
                            status = "FAILED"
                            error_msg = claim_results[0].error_message

                else:
                    raise ValueError(f"Unknown verification strategy: {verif_strat}")

            except Exception as exc:
                status = "FAILED"
                error_msg = f"Verification execution error: {type(exc).__name__}: {str(exc)}"

            verif_latency_ms = round((time.time() - t_verif_start) * 1000, 2)

            exp_rec = ExperimentRecord(
                condition_id=cond_id.value,
                source_record_id=art_id,
                language=article.language,
                prompt_strategy=strat_key,
                prompt_template_version=gen_rec.prompt_template_version,
                verification_strategy=verif_strat,
                model_id=gen_rec.model_id,
                provider=gen_rec.provider,
                generation_parameters=dict(gen_rec.generation_parameters),
                generated_summary=gen_rec.generated_summary,
                generation_latency_seconds=gen_rec.latency_seconds,
                verification_latency_ms=verif_latency_ms,
                verification_results=verification_results,
                status=status,
                error_message=error_msg,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                source_dataset=gen_rec.source_dataset,
            )

            print(f"    -> Status: {exp_rec.status} (verif_latency: {exp_rec.verification_latency_ms}ms)", flush=True)
            reverified_results.append(exp_rec)

            # Incremental save
            save_experiment_records([exp_rec], out_results_file, append=True)

    t_elapsed = round(time.time() - t_start, 2)

    # Compile Audit Report
    condition_counts: Dict[str, int] = {}
    success_counts: Dict[str, int] = {}
    failed_counts: Dict[str, int] = {}
    lang_counts: Dict[str, int] = {"en": 0, "hi": 0}

    for r in reverified_results:
        cid = r.condition_id
        condition_counts[cid] = condition_counts.get(cid, 0) + 1
        if r.status == "SUCCESS":
            success_counts[cid] = success_counts.get(cid, 0) + 1
        else:
            failed_counts[cid] = failed_counts.get(cid, 0) + 1
        lang_counts[r.language] = lang_counts.get(r.language, 0) + 1

    try:
        import torch
        torch_ver = torch.__version__
    except ImportError:
        torch_ver = "unavailable"

    try:
        import transformers
        trans_ver = transformers.__version__
    except ImportError:
        trans_ver = "unavailable"

    audit_report = {
        "timestamp_utc": start_time_utc,
        "completed_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": t_elapsed,
        "experiment_name": "FactIndic-Six-Condition-Benchmark-Experiment-Reverified",
        "regenerated_summaries": False,
        "model_id": llm_client.model_id,
        "provider": llm_client.provider,
        "nli_model_id": nli_verifier.config.model_id,
        "dependencies": {
            "torch": torch_ver,
            "transformers": trans_ver,
        },
        "source_records": {
            "english_count": len(en_records),
            "hindi_count": len(hi_records),
            "total_count": len(ordered_source_records),
        },
        "results_summary": {
            "total_evaluations": len(reverified_results),
            "expected_evaluations": len(ordered_source_records) * 6,
            "condition_breakdown": condition_counts,
            "success_breakdown": success_counts,
            "failed_breakdown": failed_counts,
            "language_distribution": lang_counts,
        },
        "output_files": {
            "results_jsonl": str(out_results_file.name),
            "audit_json": str(out_audit_file.name),
        },
    }

    with open(out_audit_file, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2, ensure_ascii=False)

    print(f"[✓] Saved reverified results to {out_results_file} ({len(reverified_results)} records)")
    print(f"[✓] Saved audit report to {out_audit_file}")
    return audit_report


def main():
    parser = argparse.ArgumentParser(description="Re-verify preserved experiment generations across six conditions.")
    parser.add_argument(
        "--results-path",
        default="data/benchmark_results/controlled_run/experiment_results.jsonl",
        help="Path to existing experiment results containing preserved summaries",
    )
    parser.add_argument(
        "--en-benchmark",
        default="data/benchmark/controlled_sample_en.jsonl",
        help="Path to English benchmark sample JSONL",
    )
    parser.add_argument(
        "--hi-benchmark",
        default="data/benchmark/controlled_sample_hi.jsonl",
        help="Path to Hindi benchmark sample JSONL",
    )
    parser.add_argument(
        "--output-dir",
        default="data/benchmark_results/controlled_run/reverified",
        help="Destination directory for reverified results and audit",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Delay in seconds between claim extraction API calls",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Rerun all conditions without resuming existing successes",
    )

    args = parser.parse_args()
    print("=" * 70)
    print("FactIndic Re-Verification Pipeline (Zero Summary Regeneration)")
    print("=" * 70)

    try:
        reverify_controlled_experiment(
            results_path=args.results_path,
            en_benchmark_path=args.en_benchmark,
            hi_benchmark_path=args.hi_benchmark,
            output_dir=args.output_dir,
            delay_seconds=args.delay,
            resume=not args.no_resume,
        )
    except Exception as exc:
        print(f"\n[X] Re-verification failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
