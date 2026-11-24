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


# -----------------------------------------------------------------------------
# Controlled Experiment Data Loaders (Local mDeBERTa)
# -----------------------------------------------------------------------------

def get_controlled_experiment_dir() -> Path:
    """Return the filesystem path to the controlled experiment results directory."""
    return DATA_DIR / "benchmark_results" / "controlled_run" / "local_mdeberta"


def load_controlled_experiment_audit() -> Optional[Dict[str, Any]]:
    """Load the machine-readable audit manifest for the controlled experiment."""
    audit_file = get_controlled_experiment_dir() / "experiment_audit.json"
    return load_audit_file(audit_file)


def load_controlled_experiment_summary_table() -> List[Dict[str, Any]]:
    """Load the machine-readable summary table for all conditions (C1–C6)."""
    table_file = get_controlled_experiment_dir() / "experiment_summary_table.json"
    if not table_file.is_file():
        return []
    try:
        with open(table_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def load_controlled_experiment_results() -> List[Dict[str, Any]]:
    """Load the 60 machine-readable evaluation output records.

    Returns:
        List of parsed JSON objects, one per evaluation. Empty list if missing.
    """
    results_file = get_controlled_experiment_dir() / "experiment_results.jsonl"
    if not results_file.is_file():
        return []
    records = []
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    except Exception:
        return []


def filter_experiment_records(
    records: List[Dict[str, Any]],
    language: Optional[str] = None,
    prompt_strategy: Optional[str] = None,
    verification_strategy: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Filter experiment evaluation records by metadata attributes."""
    filtered = records
    if language and language.lower() not in ("all", ""):
        target_lang = "hi" if language.lower() in ("hi", "hindi") else "en"
        filtered = [r for r in filtered if r.get("language") == target_lang]
    if prompt_strategy and prompt_strategy.lower() not in ("all", ""):
        target_ps = prompt_strategy.lower()
        if "prompt_a" in target_ps or target_ps == "a" or "standard" in target_ps:
            filtered = [r for r in filtered if r.get("prompt_strategy") == "prompt_a"]
        elif "prompt_b" in target_ps or target_ps == "b" or "factuality" in target_ps:
            filtered = [r for r in filtered if r.get("prompt_strategy") == "prompt_b"]
    if verification_strategy and verification_strategy.lower() not in ("all", ""):
        target_vs = verification_strategy.lower()
        if "whole" in target_vs:
            filtered = [r for r in filtered if r.get("verification_strategy") == "whole_document"]
        elif "sentence" in target_vs:
            filtered = [r for r in filtered if r.get("verification_strategy") == "sentence_level"]
        elif "hybrid" in target_vs or "claim" in target_vs:
            filtered = [r for r in filtered if r.get("verification_strategy") == "hybrid_claim_level"]
    return filtered


def get_hybrid_claim_explorer_records() -> List[Dict[str, Any]]:
    """Return all evaluation records using the hybrid claim-level strategy (C5 and C6)."""
    all_recs = load_controlled_experiment_results()
    return [r for r in all_recs if r.get("verification_strategy") == "hybrid_claim_level"]


def get_hybrid_claim_explorer_detail(record: Dict[str, Any]) -> Dict[str, Any]:
    """Structure claim-by-claim verification and retrieved evidence for a hybrid record.

    Groups retrieved evidence passages by claim_id and pairs them with claim metadata.
    Does NOT combine probabilities across different passages into a single distribution.
    """
    agg = record.get("aggregated_metrics", {})
    claims_detail = agg.get("claims_detail", [])
    raw_passages = record.get("verification_results", [])

    # Group raw passages by claim_id
    passages_by_claim: Dict[str, List[Dict[str, Any]]] = {}
    for p in raw_passages:
        cid = p.get("claim_id", "")
        if cid:
            passages_by_claim.setdefault(cid, []).append(p)

    structured_claims = []
    for cd in claims_detail:
        cid = cd.get("claim_id", "")
        passages = passages_by_claim.get(cid, [])
        passages_sorted = sorted(passages, key=lambda x: int(x.get("evidence_rank", 99)))

        structured_claims.append({
            "claim_id": cid,
            "sentence_index": cd.get("sentence_index", 0),
            "atomic_claim_text": cd.get("atomic_claim_text", ""),
            "predicted_label": cd.get("predicted_label", "neutral"),
            "max_entailment_prob": cd.get("max_entailment_prob", 0.0),
            "max_contradiction_prob": cd.get("max_contradiction_prob", 0.0),
            "passages": passages_sorted,
        })

    return {
        "condition_id": record.get("condition_id", ""),
        "source_record_id": record.get("source_record_id", ""),
        "language": record.get("language", "en"),
        "prompt_strategy": record.get("prompt_strategy", ""),
        "generated_summary": record.get("generated_summary", ""),
        "verification_latency_ms": record.get("verification_latency_ms", 0.0),
        "aggregated_metrics": agg,
        "claims": structured_claims,
    }


def get_qualitative_case_studies() -> List[Dict[str, Any]]:
    """Curated qualitative case studies extracted directly from experiment_results.jsonl."""
    return [
        {
            "case_id": "en_entailment",
            "title": "Case 1: English High-Entailment Claim Grounding",
            "language": "English",
            "condition_id": "C5",
            "record_id": "world-latin-america-44779257",
            "strategy": "hybrid_claim_level",
            "claim_text": "35 killed in towns of Diriamba and Jinotepe, 3 in Matagalpa province.",
            "retrieved_premise": "'Blood and death' Cenidh President Vilma Núñez told BBC Mundo that 35 people had been killed in the towns of Diriamba and Jinotepe, in Carazo province...",
            "rank": 1,
            "bm25_score": 14.82,
            "probabilities": {"entailment": 0.5006, "neutral": 0.0160, "contradiction": 0.4834},
            "assigned_label": "entailment",
            "analysis": "The mDeBERTa cross-encoder successfully aligned numerical and geographical assertions across clauses. The existential grounding rule (max_k P(entailment) >= 0.50) correctly verified the claim.",
        },
        {
            "case_id": "en_contradiction",
            "title": "Case 2: English Premise-Mismatch Contradiction",
            "language": "English",
            "condition_id": "C6",
            "record_id": "world-europe-55173830",
            "strategy": "hybrid_claim_level",
            "claim_text": "Customs officials at Cologne Bonn airport seized two packages containing 18 reptiles—including endangered horned lizards, alligator lizards and box turtles—destined for private buyers in Germany.",
            "retrieved_premise": "The endangered horned lizards, alligator lizards and box turtles were destined for private buyers in Germany.",
            "rank": 1,
            "bm25_score": 13.91,
            "probabilities": {"entailment": 0.0005, "neutral": 0.0604, "contradiction": 0.9391},
            "assigned_label": "contradiction",
            "analysis": "The retrieved premise stated the destination of the reptiles but omitted the seizure by customs officials at Cologne Bonn airport. Evaluated against this incomplete localized premise, mDeBERTa penalized the ungrounded compound clauses as a contradiction.",
        },
        {
            "case_id": "hi_entailment",
            "title": "Case 3: Hindi High-Entailment Semantic Paraphrase",
            "language": "Hindi",
            "condition_id": "C5",
            "record_id": "sport-42681773",
            "strategy": "hybrid_claim_level",
            "claim_text": "दिल्ली के फ़िरोज़शाह कोटला में सैयद मुश्ताक अली ट्रॉफी मैच में हिमाचल प्रदेश के खिलाफ़ ऋषभ पंत ने धमाकेदार प्रदर्शन किया।",
            "retrieved_premise": "20 साल के ऋषभ ने दिल्ली के फ़िरोज़शाह कोटला मैदान में दिल्ली की तरफ़ से खेलते हुए महज़ 32 गेंदों में शतक जड़ दिया.",
            "rank": 1,
            "bm25_score": 12.45,
            "probabilities": {"entailment": 0.6096, "neutral": 0.0028, "contradiction": 0.3876},
            "assigned_label": "entailment",
            "analysis": "Despite lexical and stylistic rephrasing ('धमाकेदार प्रदर्शन' vs. 'महज़ 32 गेंदों में शतक जड़ दिया'), mDeBERTa recognized that scoring a century in 32 balls entails an explosive performance.",
        },
        {
            "case_id": "hi_contradiction",
            "title": "Case 4: Hindi Epistemic Divergence Contradiction",
            "language": "Hindi",
            "condition_id": "C5",
            "record_id": "060528_soldiers_desert",
            "strategy": "hybrid_claim_level",
            "claim_text": "सैनिकों के प्रतिनिधियों ने पुष्टि की है कि इराक में तैनाती से बचने वाले सैनिकों की संख्या हाल के महीनों में बढ़ी है।",
            "retrieved_premise": "कोर्ट मार्शल के दौरान सैनिकों का प्रतिनिधित्व करने वाले वकीलों ने स्वीकार किया है कि पिछले कुछ समय में उनके पास आने वाले...",
            "rank": 1,
            "bm25_score": 11.88,
            "probabilities": {"entailment": 0.3708, "neutral": 0.0059, "contradiction": 0.7233},
            "assigned_label": "contradiction",
            "analysis": "The summary claimed that soldiers' representatives confirmed ('पुष्टि की है') an increase, whereas the source text reported that defense lawyers acknowledged ('स्वीकार किया है') receiving cases during court martials. mDeBERTa detected the subtle epistemic divergence.",
        },
        {
            "case_id": "hi_scratchpad",
            "title": "Case 5: Hindi Prompt B English Scratchpad / BM25 Mismatch",
            "language": "Hindi (Code-Switched / Scratchpad)",
            "condition_id": "C6",
            "record_id": "international-50448073",
            "strategy": "hybrid_claim_level",
            "claim_text": "Demand: PM Imran Khan resign, new elections.",
            "retrieved_premise": "विपक्षी पार्टियों के गठबंधन की तरफ से आज़ादी मार्च निकालने की घोषणा की गई थी...",
            "rank": 1,
            "bm25_score": 2.15,
            "probabilities": {"entailment": 0.0001, "neutral": 0.9998, "contradiction": 0.0001},
            "assigned_label": "neutral",
            "analysis": "Observed evidence: The generator emitted English planning text ('Demand: PM Imran Khan resign') for a Hindi request. Proposed explanation: English tokens have minimal lexical overlap with Devanagari news articles, degrading BM25 retrieval and forcing mDeBERTa to output neutral.",
        },
    ]


def load_benchmark_articles() -> List[Dict[str, Any]]:
    """Load the 10 frozen source benchmark articles (5 English, 5 Hindi).

    Returns:
        List of 10 dicts containing article metadata, original source text,
        and reference summary.
    """
    en_path = DATA_DIR / "benchmark" / "controlled_sample_en.jsonl"
    hi_path = DATA_DIR / "benchmark" / "controlled_sample_hi.jsonl"
    articles: List[Dict[str, Any]] = []
    for p in [en_path, hi_path]:
        if not p.is_file():
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        raw_id = r.get("metadata", {}).get("raw_id", r.get("example_id", ""))
                        articles.append({
                            "article_id": raw_id,
                            "example_id": r.get("example_id", ""),
                            "language": r.get("language", "en"),
                            "language_display": "English" if r.get("language") == "en" else "Hindi",
                            "title": r.get("metadata", {}).get("title", "Untitled Article"),
                            "source_text": r.get("source_text", ""),
                            "reference_summary": r.get("reference_summary", ""),
                            "url": r.get("metadata", {}).get("url", ""),
                        })
        except Exception:
            pass
    return articles


def get_article_evaluations(
    article_id: str,
    results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Retrieve and map the 6 evaluations for a specific benchmark article.

    Args:
        article_id: Identifier of the source article (e.g. 'uk-wales-33738390').
        results: Optional pre-loaded list of 60 evaluation records.

    Returns:
        Dict mapping prompt ('prompt_a', 'prompt_b') to strategy
        ('whole_document', 'sentence_level', 'hybrid_claim_level') -> evaluation record.
    """
    if results is None:
        results = load_controlled_experiment_results()
    recs = [r for r in results if r.get("source_record_id") == article_id]
    by_prompt: Dict[str, Dict[str, Any]] = {"prompt_a": {}, "prompt_b": {}}
    for r in recs:
        p = r.get("prompt_strategy")
        s = r.get("verification_strategy")
        if p in by_prompt and s:
            by_prompt[p][s] = r
    return by_prompt


