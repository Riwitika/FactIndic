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

# Ensure local cache directories to avoid sandbox permission errors
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("NO_TF", "1")
keras_dir = ROOT_DIR / "data" / "cache" / "keras"
keras_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("KERAS_HOME", str(keras_dir))

from factindic.cache import ResponseCache
from factindic.exceptions import QuotaExhaustedError
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
    RuleBasedClaimExtractor,
    SentenceLevelVerifier,
    WholeDocumentVerifier,
    XNLIVerifier,
    aggregate_verification_results,
)
from factindic.verification.claim_extractor import CLAIMS_JSON_SCHEMA


def reverify_controlled_experiment(
    results_path: Union[str, Path] = "data/benchmark_results/controlled_run/experiment_results.jsonl",
    en_benchmark_path: Union[str, Path] = "data/benchmark/controlled_sample_en.jsonl",
    hi_benchmark_path: Union[str, Path] = "data/benchmark/controlled_sample_hi.jsonl",
    output_dir: Union[str, Path] = "data/benchmark_results/controlled_run/reverified",
    cache_db_path: Optional[Union[str, Path]] = "data/cache/response_cache.db",
    delay_seconds: float = 3.0,
    resume: bool = True,
    dry_run: bool = False,
    nli_provider: str = "local",
    nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    claim_extractor_type: str = "rule",
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

    # 3. Initialize Response Cache and Verifiers
    cache = ResponseCache(db_path=cache_db_path) if cache_db_path else None
    from factindic.verification.config import NLIConfig
    nli_cfg = NLIConfig(provider=nli_provider, model_id=nli_model)
    nli_verifier = XNLIVerifier(config=nli_cfg, cache=cache)
    whole_doc_verifier = WholeDocumentVerifier(verifier=nli_verifier)
    sentence_verifier = SentenceLevelVerifier(verifier=nli_verifier)
    if claim_extractor_type == "rule":
        claim_extractor = RuleBasedClaimExtractor()
    else:
        llm_client = CohereClient()
        claim_extractor = AtomicClaimExtractor(client=llm_client, cache=cache, delay_seconds=delay_seconds, fallback_to_rule=True)
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

    ordered_source_records = en_records + hi_records

    if dry_run:
        remaining_count = max(0, (len(ordered_source_records) * 6) - len(completed_keys))
        print("\n" + "=" * 65)
        print("          CONTROLLED RE-VERIFICATION DRY-RUN REPORT")
        print("=" * 65)
        print(f"Total Source Records:    {len(ordered_source_records)} (5 EN, 5 HI)")
        print(f"Preserved Summaries:     {len(preserved_generations)} (10 Prompt A, 10 Prompt B)")
        est_calls = 0 if (claim_extractor_type == "rule" and nli_provider == "local") else 440
        print(f"Total Evaluations:       60 (6 conditions x 10 records)")
        print(f"Already Evaluated (SUCCESS): {len(completed_keys)}")
        print(f"Remaining Evaluations:  {remaining_count}")
        print(f"Estimated Live API Calls: {est_calls} (Claim: {claim_extractor_type}, NLI: {nli_provider})")
        print(f"Response Cache DB:       {cache_db_path}")
        if cache:
            c_stats = cache.get_stats()
            print(f"Cache Status:            {c_stats.total_entries} entries (Hits: {c_stats.hits}, Misses: {c_stats.misses})")
        print("=" * 65)
        print("[✓] Dry-run complete. Zero live API calls executed.\n")
        return {
            "dry_run": True,
            "total_records": len(ordered_source_records),
            "completed_conditions": len(completed_keys),
            "remaining_conditions": remaining_count,
            "estimated_calls": 440,
        }

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
    quota_exhausted = False

    # Iterate through the 10 source records
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

            if quota_exhausted:
                # Mark as QUOTA_BLOCKED without fabricating scores or failing blindly
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
                    verification_latency_ms=0.0,
                    verification_results=[],
                    status="QUOTA_BLOCKED",
                    error_message="Skipped: Cohere trial key monthly quota (1000 API calls) exhausted.",
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    source_dataset=gen_rec.source_dataset,
                )
                reverified_results.append(exp_rec)
                save_experiment_records(reverified_results, out_results_file, append=False)
                continue

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

            except QuotaExhaustedError as q_exc:
                quota_exhausted = True
                status = "QUOTA_BLOCKED"
                error_msg = f"QUOTA_EXHAUSTED: {q_exc}"
                print(f"    [!] Monthly API Quota Exhausted: {q_exc}. Halting calls.", flush=True)

            except Exception as exc:
                err_str = str(exc).lower()
                if "1000 api calls / month" in err_str or ("trial key" in err_str and "429" in err_str):
                    quota_exhausted = True
                    status = "QUOTA_BLOCKED"
                    error_msg = f"QUOTA_EXHAUSTED: {exc}"
                    print(f"    [!] Monthly API Quota Exhausted: {exc}. Halting calls.", flush=True)
                else:
                    status = "FAILED"
                    error_msg = f"Verification execution error: {type(exc).__name__}: {str(exc)}"

            verif_latency_ms = round((time.time() - t_verif_start) * 1000, 2)

            agg_metrics = (
                aggregate_verification_results(verif_strat, verification_results)
                if status == "SUCCESS"
                else None
            )

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
                aggregated_metrics=agg_metrics,
            )

            print(f"    -> Status: {exp_rec.status} (verif_latency: {exp_rec.verification_latency_ms}ms)", flush=True)
            reverified_results.append(exp_rec)

            # Incremental clean save of all verified records so far
            save_experiment_records(reverified_results, out_results_file, append=False)

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

    model_gen_id = next((r.model_id for r in preserved_generations.values() if r.model_id), "command-a-plus-05-2026")
    provider_gen = next((r.provider for r in preserved_generations.values() if r.provider), "cohere")

    # Generate Structured Summary Tables
    summary_rows: List[Dict[str, Any]] = []
    for cond_id_val in ["C1", "C2", "C3", "C4", "C5", "C6"]:
        cond_records = [r for r in reverified_results if r.condition_id == cond_id_val]
        if not cond_records:
            continue
        c_def = CONDITION_DEFINITIONS[ConditionID(cond_id_val)]
        strat_name = c_def["prompt_strategy"].value
        verif_strat_name = c_def["verification_strategy"]

        for lang_filter in ["en", "hi", "all"]:
            subset = cond_records if lang_filter == "all" else [r for r in cond_records if r.language == lang_filter]
            if not subset:
                continue

            success_sub = [r for r in subset if r.status == "SUCCESS" and r.aggregated_metrics]
            count_total = len(subset)
            count_success = len(success_sub)

            # Compute macro metrics
            entail_scores: List[float] = []
            entail_ratios: List[float] = []
            contra_ratios: List[float] = []
            neut_ratios: List[float] = []

            for r in success_sub:
                agg = r.aggregated_metrics or {}
                if verif_strat_name == "whole_document":
                    entail_scores.append(float(agg.get("entailment_prob", 0.0)))
                    is_ent = 1.0 if agg.get("predicted_label") == "entailment" else 0.0
                    is_contra = 1.0 if agg.get("predicted_label") == "contradiction" else 0.0
                    is_neut = 1.0 if agg.get("predicted_label") == "neutral" else 0.0
                    entail_ratios.append(is_ent)
                    contra_ratios.append(is_contra)
                    neut_ratios.append(is_neut)
                elif verif_strat_name == "sentence_level":
                    entail_scores.append(float(agg.get("mean_sentence_entailment", 0.0)))
                    entail_ratios.append(float(agg.get("entailment_ratio", 0.0)))
                    contra_ratios.append(float(agg.get("contradiction_ratio", 0.0)))
                    neut_ratios.append(float(agg.get("neutral_ratio", 0.0)))
                elif verif_strat_name == "hybrid_claim_level":
                    entail_scores.append(float(agg.get("mean_claim_entailment", 0.0)))
                    entail_ratios.append(float(agg.get("entailment_ratio", 0.0)))
                    contra_ratios.append(float(agg.get("contradiction_ratio", 0.0)))
                    neut_ratios.append(float(agg.get("neutral_ratio", 0.0)))

            mean_lat = round(sum(r.verification_latency_ms for r in subset) / count_total, 2) if count_total else 0.0
            n_sub = len(success_sub)
            summary_rows.append({
                "condition_id": cond_id_val,
                "prompt_strategy": strat_name,
                "verification_strategy": verif_strat_name,
                "language": lang_filter,
                "total_evaluations": count_total,
                "success_evaluations": count_success,
                "macro_mean_entailment_score": round(sum(entail_scores) / n_sub, 4) if n_sub else 0.0,
                "macro_entailment_ratio": round(sum(entail_ratios) / n_sub, 4) if n_sub else 0.0,
                "macro_contradiction_ratio": round(sum(contra_ratios) / n_sub, 4) if n_sub else 0.0,
                "macro_neutral_ratio": round(sum(neut_ratios) / n_sub, 4) if n_sub else 0.0,
                "mean_latency_ms": mean_lat,
            })

    out_summary_json = out_dir / "experiment_summary_table.json"
    with open(out_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    out_summary_csv = out_dir / "experiment_summary_table.csv"
    if summary_rows:
        keys = list(summary_rows[0].keys())
        with open(out_summary_csv, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for row in summary_rows:
                f.write(",".join(str(row[k]) for k in keys) + "\n")

    quota_blocked_count = sum(1 for r in reverified_results if r.status == "QUOTA_BLOCKED")
    audit_report = {
        "timestamp_utc": start_time_utc,
        "completed_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": t_elapsed,
        "experiment_name": "FactIndic-Six-Condition-Benchmark-Experiment-Reverified",
        "regenerated_summaries": False,
        "generation_model_id": model_gen_id,
        "generation_provider": provider_gen,
        "nli_model_id": nli_verifier.config.model_id,
        "nli_model_revision": getattr(nli_verifier.config, "model_revision", "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"),
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
            "quota_blocked_count": quota_blocked_count,
            "language_distribution": lang_counts,
        },
        "cache_stats": cache.get_stats().to_dict() if cache else None,
        "output_files": {
            "results_jsonl": str(out_results_file.name),
            "audit_json": str(out_audit_file.name),
            "summary_table_json": str(out_summary_json.name),
            "summary_table_csv": str(out_summary_csv.name),
        },
    }

    # Final clean rewrite of all reverified records to ensure exact ordering and no duplicates
    save_experiment_records(reverified_results, out_results_file, append=False)

    with open(out_audit_file, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2, ensure_ascii=False)

    print(f"[✓] Saved reverified results to {out_results_file} ({len(reverified_results)} records)")
    print(f"[✓] Saved summary table to {out_summary_json} and {out_summary_csv}")
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
        default="data/benchmark_results/controlled_run/local_mdeberta",
        help="Destination directory for reverified results and audit",
    )
    parser.add_argument(
        "--cache-db",
        default="data/cache/response_cache.db",
        help="Path to SQLite persistent response cache DB (default: data/cache/response_cache.db)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Delay in seconds between claim extraction API calls",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate estimated API calls and audit state without executing API calls",
    )
    parser.add_argument(
        "--claim-extractor",
        default="rule",
        choices=["rule", "llm"],
        help="Claim extraction strategy ('rule' for deterministic rule-based extractor, 'llm' for Cohere)",
    )
    parser.add_argument(
        "--nli-provider",
        default="local",
        choices=["local", "cohere", "hosted"],
        help="Inference provider for NLI verification",
    )
    parser.add_argument(
        "--nli-model",
        default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        help="Model ID for NLI verification",
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
            cache_db_path=args.cache_db,
            delay_seconds=args.delay,
            resume=not args.no_resume,
            dry_run=args.dry_run,
            nli_provider=args.nli_provider,
            nli_model=args.nli_model,
            claim_extractor_type=args.claim_extractor,
        )
    except Exception as exc:
        print(f"\n[X] Re-verification failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
