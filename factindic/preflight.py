"""Preflight Verification and API Call Budget Estimation.

Performs comprehensive validation of benchmark manifests, cryptographic file integrity,
API credentials, and resource budgets before any generation or verification API calls are made.
Prevents partial execution failures and quota exhaustion during full-scale experiments.
"""

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

from factindic.benchmark_manifest import validate_benchmark_manifest
from factindic.models.alternative import load_local_factindic_env


COHERE_TRIAL_MONTHLY_LIMIT = 1000
COHERE_TRIAL_RPM_LIMIT = 20
COHERE_PRODUCTION_RPM_DEFAULT = 100


@dataclass
class CallBudgetEstimate:
    """Detailed breakdown of estimated API calls and runtime."""

    num_records: int
    prompts_per_record: int
    generation_calls: int
    whole_doc_nli_calls: int
    avg_sentences_per_summary: float
    sentence_nli_calls: int
    claim_extraction_calls: int
    avg_claims_per_sentence: float
    top_k_evidence: int
    hybrid_claim_nli_calls: int
    total_calls: int
    estimated_minutes_trial: float
    estimated_hours_trial: float
    estimated_minutes_production: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def estimate_api_calls(
    num_records: int = 200,
    prompts_per_record: int = 2,
    avg_sentences_per_summary: float = 3.4,
    avg_claims_per_sentence: float = 2.088,
    top_k_evidence: int = 3,
) -> CallBudgetEstimate:
    """Calculate the estimated API calls and execution time for an experiment run.

    Default empirical averages are derived from the 20-summary XL-Sum controlled run:
    - 3.4 sentences per generated summary
    - ~2.088 atomic claims per sentence (~7.1 claims per summary)
    - 3 evidence passages retrieved per claim

    Args:
        num_records: Total number of records (e.g. 200 for 100 EN + 100 HI).
        prompts_per_record: Summaries generated per record (Prompt A and Prompt B = 2).
        avg_sentences_per_summary: Mean sentence count in summaries.
        avg_claims_per_sentence: Mean claims extracted per summary sentence.
        top_k_evidence: Number of candidate evidence passages retrieved per claim.

    Returns:
        CallBudgetEstimate with full breakdown.
    """
    total_summaries = num_records * prompts_per_record
    generation_calls = total_summaries
    whole_doc_nli_calls = total_summaries

    total_sentences = int(round(total_summaries * avg_sentences_per_summary))
    sentence_nli_calls = total_sentences
    claim_extraction_calls = total_sentences

    total_claims = int(round(total_sentences * avg_claims_per_sentence))
    hybrid_claim_nli_calls = total_claims * top_k_evidence

    total_calls = (
        generation_calls
        + whole_doc_nli_calls
        + sentence_nli_calls
        + claim_extraction_calls
        + hybrid_claim_nli_calls
    )

    minutes_trial = total_calls / COHERE_TRIAL_RPM_LIMIT
    hours_trial = minutes_trial / 60.0
    minutes_prod = total_calls / COHERE_PRODUCTION_RPM_DEFAULT

    return CallBudgetEstimate(
        num_records=num_records,
        prompts_per_record=prompts_per_record,
        generation_calls=generation_calls,
        whole_doc_nli_calls=whole_doc_nli_calls,
        avg_sentences_per_summary=avg_sentences_per_summary,
        sentence_nli_calls=sentence_nli_calls,
        claim_extraction_calls=claim_extraction_calls,
        avg_claims_per_sentence=avg_claims_per_sentence,
        top_k_evidence=top_k_evidence,
        hybrid_claim_nli_calls=hybrid_claim_nli_calls,
        total_calls=total_calls,
        estimated_minutes_trial=round(minutes_trial, 1),
        estimated_hours_trial=round(hours_trial, 2),
        estimated_minutes_production=round(minutes_prod, 1),
    )


def check_api_credentials() -> Dict[str, Any]:
    """Verify presence and configuration of Cohere API credentials.

    Does NOT expose or print the actual secret key.

    Returns:
        Dictionary with credential status.
    """
    local_env = load_local_factindic_env()
    key = (
        os.environ.get("COHERE_API_KEY")
        or os.environ.get("CO_API_KEY")
        or local_env.get("COHERE_API_KEY")
        or local_env.get("CO_API_KEY")
    )

    if not key or not key.strip():
        return {
            "configured": False,
            "key_present": False,
            "key_length": 0,
            "tier_hint": "missing",
            "message": "Cohere API key not found in environment or .factindic_env.",
        }

    key_len = len(key.strip())
    # Cohere trial keys are standard 40-character API keys
    tier_hint = "trial_or_production"
    return {
        "configured": True,
        "key_present": True,
        "key_length": key_len,
        "tier_hint": tier_hint,
        "message": f"Cohere API key configured ({key_len} chars).",
    }


def run_preflight(
    manifest_path: Union[str, Path] = "data/benchmark/benchmark_manifest.json",
    benchmark_dir: Optional[Union[str, Path]] = None,
    target_records: Optional[int] = None,
    allow_insufficient_quota: bool = False,
) -> Dict[str, Any]:
    """Execute full preflight verification.

    Args:
        manifest_path: Path to benchmark_manifest.json.
        benchmark_dir: Optional directory containing benchmark files.
        target_records: Optional record count override (e.g. 10 for controlled test).
        allow_insufficient_quota: If True, warns rather than fails on quota deficit.

    Returns:
        Structured preflight report dictionary.
    """
    m_path = Path(manifest_path)
    b_dir = Path(benchmark_dir) if benchmark_dir else m_path.parent

    # 1. Manifest and file integrity check
    manifest_valid, manifest_errors = validate_benchmark_manifest(
        manifest_path=m_path, benchmark_dir=b_dir
    )

    # 2. Disk space check
    total_b, used_b, free_b = shutil.disk_usage(Path.cwd())
    free_gb = free_b / (1024**3)

    # 3. Determine record count
    if target_records is not None:
        rec_count = target_records
    elif manifest_valid:
        import json
        with open(m_path, "r", encoding="utf-8") as f:
            m_data = json.load(f)
        rec_count = m_data.get("total_records", 200)
    else:
        rec_count = 200

    # 4. Estimate calls
    budget = estimate_api_calls(num_records=rec_count)

    # 5. Check credentials
    creds = check_api_credentials()

    # 6. Assess quota feasibility
    can_proceed = True
    blockers: List[str] = []

    if not manifest_valid:
        can_proceed = False
        blockers.extend(manifest_errors)

    if not creds["configured"]:
        can_proceed = False
        blockers.append("Cohere API key is not configured.")

    if free_gb < 1.0:
        can_proceed = False
        blockers.append(f"Insufficient disk space ({free_gb:.2f} GB free, requires >= 1.0 GB).")

    # If running full benchmark on a trial key, budget (12,040) vastly exceeds monthly quota (1,000)
    if budget.total_calls > COHERE_TRIAL_MONTHLY_LIMIT and not allow_insufficient_quota:
        can_proceed = False
        blockers.append(
            f"Required API calls ({budget.total_calls:,}) exceed the Cohere trial key monthly cap "
            f"({COHERE_TRIAL_MONTHLY_LIMIT:,} calls). Execution requires a Production API key "
            f"or local GPU inference to prevent mid-run quota exhaustion."
        )

    return {
        "status": "READY" if can_proceed else "BLOCKED",
        "can_proceed": can_proceed,
        "blockers": blockers,
        "manifest": {
            "path": str(m_path),
            "is_valid": manifest_valid,
            "errors": manifest_errors,
        },
        "credentials": creds,
        "budget": budget.to_dict(),
        "storage": {
            "free_disk_gb": round(free_gb, 2),
            "is_sufficient": free_gb >= 1.0,
        },
    }
