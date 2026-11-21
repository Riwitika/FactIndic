"""Helper functions for FactIndic Streamlit research demo website."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from factindic.datasets.xlsum import XLSumLoader
from factindic.generation import GenerationRecord, load_generation_records
from factindic.prompts.templates import PROMPT_TEMPLATE_VERSION, PromptStrategy, build_summarization_prompt
from factindic.schema import FactIndicRecord
from factindic.validation import validate_record

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

TARGET_RECORD_IDS: Dict[str, List[str]] = {
    "english": ["world-us-canada-38697908", "uk-wales-politics-48586353"],
    "hindi": ["081027_sadhvi_test_pp", "070929_pakistan_protest"],
}

TARGET_TITLES: Dict[str, str] = {
    "world-us-canada-38697908": "Trump inauguration speech: 'Angry', 'authentic', 'primal'",
    "uk-wales-politics-48586353": "NHS Wales: Court action if trade deals affect service?",
    "081027_sadhvi_test_pp": "साध्वी का ब्रेन मैपिंग और नॉर्को टेस्ट",
    "070929_pakistan_protest": "मुशर्रफ़ का पर्चा मंज़ूर, प्रदर्शनों पर लाठी",
}


def load_audit_file(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Safely load a JSON audit file.

    Args:
        file_path: Path to the audit JSON file.

    Returns:
        Parsed dict, or None if the file does not exist or is invalid.
    """
    path = Path(file_path)
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_project_audits() -> Dict[str, Optional[Dict[str, Any]]]:
    """Retrieve all standard FactIndic audit reports.

    Returns:
        Dictionary mapping audit keys to their parsed contents.
    """
    return {
        "live_xlsum": load_audit_file(DATA_DIR / "audits" / "live_xlsum_verification_audit.json"),
        "demo_generation": load_audit_file(DATA_DIR / "generations" / "factindic_demo_only_audit.json"),
        "smoke_generation": load_audit_file(DATA_DIR / "generations" / "generation_experiment_audit.json"),
        "controlled_generation": load_audit_file(DATA_DIR / "generations" / "controlled_generation_audit.json"),
    }


def fetch_live_target_records(
    language: str,
    max_scan_samples: Optional[int] = None,
) -> Tuple[List[FactIndicRecord], Optional[str]]:
    """Retrieve and validate target XL-Sum records from Hugging Face for the given language.

    Args:
        language: 'english' or 'hindi'.
        max_scan_samples: Maximum number of records to scan in the stream.
            Defaults to None (scans until all target IDs are matched or stream ends).

    Returns:
        Tuple of (list of matched valid FactIndicRecords, optional error message string).
    """
    lang_key = "hindi" if language.lower() in ("hi", "hindi") else "english"
    target_ids = set(TARGET_RECORD_IDS.get(lang_key, []))
    matched_records: List[FactIndicRecord] = []

    try:
        loader = XLSumLoader(language=lang_key)
        for raw_item, record in loader.stream_raw(split="test", max_samples=max_scan_samples):
            raw_id = raw_item.get("id")
            if raw_id in target_ids:
                is_valid, reason = validate_record(record)
                if not is_valid:
                    return [], f"Record '{raw_id}' failed schema validation: {reason}"
                matched_records.append(record)
                if len(matched_records) >= len(target_ids):
                    break
        return matched_records, None
    except Exception as exc:
        return [], str(exc)


def load_demo_generations() -> List[GenerationRecord]:
    """Load existing demo generation records if available on disk."""
    results_path = DATA_DIR / "generations" / "factindic_demo_only_results.jsonl"
    if not results_path.is_file():
        return []
    try:
        return load_generation_records(results_path)
    except Exception:
        return []


def get_pipeline_stages() -> List[Dict[str, Any]]:
    """Return structured specifications for the FactIndic research pipeline."""
    return [
        {
            "stage_number": 1,
            "title": "Dataset Curation & Validation",
            "status": "Verified (Live)",
            "status_type": "success",
            "description": "Stream genuine articles and reference summaries from csebuetnlp/xlsum for English and Hindi. Normalizes text to Unicode NFC and applies strict schema validation.",
            "implementation": "factindic.datasets.xlsum.XLSumLoader",
        },
        {
            "stage_number": 2,
            "title": "Dual Summarization Prompting",
            "status": "Verified (Templates)",
            "status_type": "success",
            "description": "Compare Prompt A (Standard Abstractive) vs. Prompt B (Factuality-Aware with explicit negative constraints on hallucinations, unsupported facts, and entity distortions).",
            "implementation": "factindic.prompts.templates.build_summarization_prompt",
        },
        {
            "stage_number": 3,
            "title": "Model Inference",
            "status": "Verified (20/20 Generated)",
            "status_type": "success",
            "description": "Executed Prompt A and Prompt B across 10 genuine XL-Sum records (5 English, 5 Hindi) using Cohere Command A+ (command-a-plus-05-2026). All 20 summaries verified.",
            "implementation": "factindic.controlled_experiment / factindic.models.aya",
        },
        {
            "stage_number": 4,
            "title": "Verification Approach 1: Whole-Doc NLI",
            "status": "Planned",
            "status_type": "info",
            "description": "Score full summary against the complete document premise using multilingual NLI cross-encoders (e.g. XLM-RoBERTa / IndicBERT-NLI).",
            "implementation": "Upcoming Phase 4",
        },
        {
            "stage_number": 5,
            "title": "Verification Approach 2: Claim-Level NLI",
            "status": "Planned",
            "status_type": "info",
            "description": "Decompose generated summaries into atomic factual claims, retrieve the most relevant source sentences as evidence, and evaluate entailment locally.",
            "implementation": "Upcoming Phase 4",
        },
        {
            "stage_number": 6,
            "title": "Verification Approach 3: Hybrid Factuality Check",
            "status": "Planned",
            "status_type": "info",
            "description": "Combine claim-level NLI with targeted entity alignment, numerical/date consistency, negation cross-checks, and relational dependency parsing.",
            "implementation": "Upcoming Phase 4",
        },
        {
            "stage_number": 7,
            "title": "Human Evaluation & FRANK Benchmarking",
            "status": "Planned (Protocol Defined)",
            "status_type": "info",
            "description": "Double-blind expert human annotation on 200 summaries across 7 FRANK error categories (PredE, EntE, CircE, CorefE, LinkE, OutE, OthE) to compute Cohen's kappa and benchmark NLI metrics.",
            "implementation": "docs/human_annotation_guide.md",
        },
    ]
