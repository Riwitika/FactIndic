"""Unified Six-Condition Experiment Pipeline for FactIndic.

Orchestrates a 2 x 3 factorial experimental design across identical source records:
    - 2 Prompting Strategies: Prompt A (standard) and Prompt B (factuality-aware)
    - 3 Verification Approaches: Whole-Document NLI, Sentence-Level NLI, and Hybrid Claim-Level NLI

Conditions Matrix:
    - C1: Prompt A + Whole-Document NLI
    - C2: Prompt B + Whole-Document NLI
    - C3: Prompt A + Sentence-Level NLI
    - C4: Prompt B + Sentence-Level NLI
    - C5: Prompt A + Hybrid Claim-Level NLI
    - C6: Prompt B + Hybrid Claim-Level NLI

Methodology & Experimental Controls:
    - Source Records: Identical source records evaluated across all 6 conditions.
    - Generation Parity: Cohere Command A+ with identical generation parameters.
    - Independent Reporting: English and Hindi evaluated and reported separately.
    - Traceability & Resumption: Resumes without re-running completed records.
    - Honest Failure Recording: API/model failures are recorded explicitly without false success.
    - Epistemic Safeguards: No synthetic composite scores; raw probabilities and claim outputs are preserved.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from factindic.generation import (
    GenerationRecord,
    run_article_generation,
    save_generation_records,
)
from factindic.models.aya import CohereClient
from factindic.models.base import BaseLLMClient
from factindic.prompts.templates import PROMPT_TEMPLATE_VERSION, PromptStrategy
from factindic.schema import FactIndicRecord
from factindic.verification import (
    AtomicClaimExtractor,
    BM25EvidenceRetriever,
    ClaimEvidenceResult,
    ClaimVerificationResult,
    DocumentVerificationResult,
    ExtractedClaim,
    HybridClaimVerifier,
    SentenceLevelVerifier,
    SentenceVerificationResult,
    WholeDocumentVerifier,
    XNLIVerifier,
)
from factindic.verification.nli import SUPPORTED_VERIFICATION_LANGUAGES

EXPERIMENT_MODEL_ID = "command-a-plus-05-2026"
EXPERIMENT_PROVIDER = "cohere"
EXPERIMENT_GENERATION_PARAMS: Dict[str, Any] = {
    "temperature": 0.3,
    "max_new_tokens": 256,
    "top_p": 0.9,
}


class ConditionID(str, Enum):
    """The six experimental conditions in the FactIndic 2x3 matrix."""

    C1 = "C1"  # Prompt A + Whole-Document NLI
    C2 = "C2"  # Prompt B + Whole-Document NLI
    C3 = "C3"  # Prompt A + Sentence-Level NLI
    C4 = "C4"  # Prompt B + Sentence-Level NLI
    C5 = "C5"  # Prompt A + Hybrid Claim-Level NLI
    C6 = "C6"  # Prompt B + Hybrid Claim-Level NLI


CONDITION_SPECS: Dict[ConditionID, Dict[str, Any]] = {
    ConditionID.C1: {
        "prompt_strategy": PromptStrategy.PROMPT_A,
        "verification_strategy": "whole_document",
        "description": "Prompt A (Standard) with Whole-Document NLI verification",
    },
    ConditionID.C2: {
        "prompt_strategy": PromptStrategy.PROMPT_B,
        "verification_strategy": "whole_document",
        "description": "Prompt B (Factuality-Aware) with Whole-Document NLI verification",
    },
    ConditionID.C3: {
        "prompt_strategy": PromptStrategy.PROMPT_A,
        "verification_strategy": "sentence_level",
        "description": "Prompt A (Standard) with Sentence-Level NLI verification",
    },
    ConditionID.C4: {
        "prompt_strategy": PromptStrategy.PROMPT_B,
        "verification_strategy": "sentence_level",
        "description": "Prompt B (Factuality-Aware) with Sentence-Level NLI verification",
    },
    ConditionID.C5: {
        "prompt_strategy": PromptStrategy.PROMPT_A,
        "verification_strategy": "hybrid_claim_level",
        "description": "Prompt A (Standard) with Hybrid Claim-Level NLI verification",
    },
    ConditionID.C6: {
        "prompt_strategy": PromptStrategy.PROMPT_B,
        "verification_strategy": "hybrid_claim_level",
        "description": "Prompt B (Factuality-Aware) with Hybrid Claim-Level NLI verification",
    },
}


@dataclass
class ExperimentRecord:
    """Standardized record of a single condition run on a source article.

    Attributes:
        condition_id: Condition identifier ('C1' through 'C6').
        source_record_id: Identifier of the source article (raw_id or example_id).
        language: Language code ('en' or 'hi').
        prompt_strategy: Prompting strategy ('prompt_a' or 'prompt_b').
        prompt_template_version: Version string of prompt templates.
        verification_strategy: Strategy ('whole_document', 'sentence_level', or 'hybrid_claim_level').
        model_id: Generation LLM identifier.
        provider: Generation LLM provider.
        generation_parameters: Hyperparameter dictionary.
        generated_summary: Summary text produced by the model.
        generation_latency_seconds: LLM generation execution latency.
        verification_latency_ms: Verification execution latency in ms.
        verification_results: List of raw verification result dictionaries.
        status: Status code ('SUCCESS' or 'FAILED').
        error_message: Optional error message if generation or verification failed.
        timestamp_utc: Execution timestamp in ISO format.
        source_dataset: Originating benchmark dataset name.
    """

    condition_id: str
    source_record_id: str
    language: str
    prompt_strategy: str
    prompt_template_version: str
    verification_strategy: str
    model_id: str
    provider: str
    generation_parameters: Dict[str, Any]
    generated_summary: str
    generation_latency_seconds: float
    verification_latency_ms: float
    verification_results: List[Dict[str, Any]]
    status: str = "SUCCESS"
    error_message: Optional[str] = None
    timestamp_utc: str = ""
    source_dataset: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment record to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRecord":
        """Construct ExperimentRecord from dictionary."""
        return cls(
            condition_id=str(data.get("condition_id", "")),
            source_record_id=str(data.get("source_record_id", "")),
            language=str(data.get("language", "")),
            prompt_strategy=str(data.get("prompt_strategy", "")),
            prompt_template_version=str(data.get("prompt_template_version", "")),
            verification_strategy=str(data.get("verification_strategy", "")),
            model_id=str(data.get("model_id", "")),
            provider=str(data.get("provider", "")),
            generation_parameters=dict(data.get("generation_parameters", {})),
            generated_summary=str(data.get("generated_summary", "")),
            generation_latency_seconds=float(data.get("generation_latency_seconds", 0.0)),
            verification_latency_ms=float(data.get("verification_latency_ms", 0.0)),
            verification_results=list(data.get("verification_results", [])),
            status=str(data.get("status", "SUCCESS")),
            error_message=data.get("error_message"),
            timestamp_utc=str(data.get("timestamp_utc", "")),
            source_dataset=data.get("source_dataset"),
        )


def _get_article_id(article: Any) -> str:
    """Helper to extract canonical source record identifier."""
    if hasattr(article, "metadata") and isinstance(article.metadata, dict):
        if "raw_id" in article.metadata and article.metadata["raw_id"]:
            return str(article.metadata["raw_id"])
    if hasattr(article, "source_record_id") and getattr(article, "source_record_id"):
        return str(getattr(article, "source_record_id"))
    if hasattr(article, "example_id") and getattr(article, "example_id"):
        return str(getattr(article, "example_id"))
    if hasattr(article, "id") and getattr(article, "id"):
        return str(getattr(article, "id"))
    return "unknown_record"


class UnifiedExperimentRunner:
    """Orchestrates the 6-condition experiment across identical source articles."""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        nli_verifier: Optional[XNLIVerifier] = None,
        whole_doc_verifier: Optional[WholeDocumentVerifier] = None,
        sentence_verifier: Optional[SentenceLevelVerifier] = None,
        claim_extractor: Optional[AtomicClaimExtractor] = None,
        evidence_retriever: Optional[BM25EvidenceRetriever] = None,
        hybrid_verifier: Optional[HybridClaimVerifier] = None,
        generation_params: Optional[Dict[str, Any]] = None,
        delay_seconds: float = 0.5,
    ) -> None:
        """Initialize experiment runner with all necessary components.

        Reuses existing verifiers and LLM clients without duplicating logic.
        """
        self.llm_client = llm_client or CohereClient()
        self.nli_verifier = nli_verifier or XNLIVerifier()
        self.whole_doc_verifier = whole_doc_verifier or WholeDocumentVerifier(verifier=self.nli_verifier)
        self.sentence_verifier = sentence_verifier or SentenceLevelVerifier(verifier=self.nli_verifier)
        self.claim_extractor = claim_extractor or AtomicClaimExtractor(
            client=self.llm_client,
            delay_seconds=delay_seconds,
        )
        self.evidence_retriever = evidence_retriever or BM25EvidenceRetriever()
        self.hybrid_verifier = hybrid_verifier or HybridClaimVerifier(
            verifier=self.nli_verifier,
            retriever=self.evidence_retriever,
        )
        self.generation_params = generation_params or dict(EXPERIMENT_GENERATION_PARAMS)
        self.delay_seconds = delay_seconds

    def _generate_or_reuse(
        self,
        article: FactIndicRecord,
        strategy: PromptStrategy,
        generation_cache: Dict[Tuple[str, str], GenerationRecord],
    ) -> GenerationRecord:
        """Generate summary or retrieve from generation cache to ensure exact parameter parity."""
        art_id = _get_article_id(article)
        strat_key = strategy.value
        cache_key = (art_id, strat_key)

        if cache_key in generation_cache:
            return generation_cache[cache_key]

        gen_rec = run_article_generation(
            record=article,
            client=self.llm_client,
            strategy=strategy,
            temperature=self.generation_params["temperature"],
            max_new_tokens=self.generation_params["max_new_tokens"],
            top_p=self.generation_params["top_p"],
        )
        generation_cache[cache_key] = gen_rec
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
        return gen_rec

    def run_record_condition(
        self,
        article: FactIndicRecord,
        condition_id: ConditionID,
        generation_cache: Dict[Tuple[str, str], GenerationRecord],
    ) -> ExperimentRecord:
        """Run a single experimental condition on an individual source article.

        Args:
            article: Source FactIndicRecord instance.
            condition_id: ConditionID (C1 to C6).
            generation_cache: Shared dictionary caching generations by (source_id, prompt_strategy).

        Returns:
            Structured ExperimentRecord.
        """
        spec = CONDITION_SPECS[condition_id]
        strat = spec["prompt_strategy"]
        verif_strat = spec["verification_strategy"]
        art_id = _get_article_id(article)
        lang = getattr(article, "language", "")
        timestamp = datetime.now(timezone.utc).isoformat()

        # Step 1: Generate or reuse summary under the condition's prompt strategy
        gen_rec = self._generate_or_reuse(article, strat, generation_cache)

        # If generation failed, record experiment failure honestly
        if gen_rec.status != "SUCCESS" or not gen_rec.generated_summary.strip():
            return ExperimentRecord(
                condition_id=condition_id.value,
                source_record_id=art_id,
                language=lang,
                prompt_strategy=strat.value,
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
                verification_strategy=verif_strat,
                model_id=gen_rec.model_id or EXPERIMENT_MODEL_ID,
                provider=gen_rec.provider or EXPERIMENT_PROVIDER,
                generation_parameters=self.generation_params,
                generated_summary="",
                generation_latency_seconds=gen_rec.latency_seconds,
                verification_latency_ms=0.0,
                verification_results=[],
                status="FAILED",
                error_message=f"Generation failed: {gen_rec.error_message or 'Empty summary generated.'}",
                timestamp_utc=timestamp,
                source_dataset=getattr(article, "metadata", {}).get("source_dataset", "csebuetnlp/xlsum"),
            )

        # Step 2: Route to the appropriate verification approach
        t_verif_start = time.time()
        verification_results: List[Dict[str, Any]] = []
        status = "SUCCESS"
        error_msg = None

        try:
            if verif_strat == "whole_document":
                doc_res = self.whole_doc_verifier.verify_record(gen_rec)
                verification_results = [doc_res.to_dict()]
                if doc_res.status == "FAILED":
                    status = "FAILED"
                    error_msg = doc_res.error_message

            elif verif_strat == "sentence_level":
                sent_results = self.sentence_verifier.verify_record(gen_rec)
                verification_results = [r.to_dict() for r in sent_results]
                if any(r.status == "FAILED" for r in sent_results):
                    # Check if all failed
                    if all(r.status == "FAILED" for r in sent_results):
                        status = "FAILED"
                        error_msg = sent_results[0].error_message

            elif verif_strat == "hybrid_claim_level":
                # Extract claims
                claims = self.claim_extractor.extract_from_record(gen_rec)
                # Verify claims using hybrid verifier
                claim_results = self.hybrid_verifier.verify_record(gen_rec, claims=claims)
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

        return ExperimentRecord(
            condition_id=condition_id.value,
            source_record_id=art_id,
            language=lang,
            prompt_strategy=strat.value,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            verification_strategy=verif_strat,
            model_id=gen_rec.model_id or EXPERIMENT_MODEL_ID,
            provider=gen_rec.provider or EXPERIMENT_PROVIDER,
            generation_parameters=self.generation_params,
            generated_summary=gen_rec.generated_summary,
            generation_latency_seconds=gen_rec.latency_seconds,
            verification_latency_ms=verif_latency_ms,
            verification_results=verification_results,
            status=status,
            error_message=error_msg,
            timestamp_utc=timestamp,
            source_dataset=gen_rec.source_dataset,
        )

    def run_all_conditions(
        self,
        records: Sequence[FactIndicRecord],
        output_path: Optional[Union[str, Path]] = None,
        resume: bool = True,
    ) -> List[ExperimentRecord]:
        """Execute all six conditions (C1 to C6) across the provided source records.

        Resumable: skips already successfully completed (source_record_id, condition_id) pairs.

        Args:
            records: Sequence of FactIndicRecord instances.
            output_path: Optional destination file path to save incremental JSONL records.
            resume: Whether to load existing results and skip already completed runs.

        Returns:
            List of all ExperimentRecord results.
        """
        out_file = Path(output_path) if output_path else None
        completed_records: Dict[Tuple[str, str], ExperimentRecord] = {}

        # 1. Resumption check
        if resume and out_file and out_file.exists():
            try:
                loaded = load_experiment_records(out_file)
                for r in loaded:
                    if r.status == "SUCCESS":
                        completed_records[(r.source_record_id, r.condition_id)] = r
            except Exception:
                pass

        # Shared generation cache across conditions to avoid regenerating
        generation_cache: Dict[Tuple[str, str], GenerationRecord] = {}

        all_conditions = [
            ConditionID.C1,
            ConditionID.C2,
            ConditionID.C3,
            ConditionID.C4,
            ConditionID.C5,
            ConditionID.C6,
        ]

        results: List[ExperimentRecord] = []

        for art_idx, article in enumerate(records, 1):
            art_id = _get_article_id(article)
            for cond_id in all_conditions:
                key = (art_id, cond_id.value)
                if key in completed_records:
                    results.append(completed_records[key])
                    continue

                print(
                    f"[{art_idx}/{len(records)}] Evaluating {cond_id.value} for {art_id} ({article.language})...",
                    flush=True,
                )
                exp_rec = self.run_record_condition(
                    article=article,
                    condition_id=cond_id,
                    generation_cache=generation_cache,
                )
                print(
                    f"    -> Status: {exp_rec.status} (verif_latency: {exp_rec.verification_latency_ms}ms)",
                    flush=True,
                )
                results.append(exp_rec)

                # Incremental persistence if output_path is configured
                if out_file:
                    save_experiment_records([exp_rec], out_file, append=True)

        return results


def save_experiment_records(
    records: Sequence[ExperimentRecord],
    file_path: Union[str, Path],
    append: bool = False,
) -> None:
    """Save ExperimentRecord objects to a JSONL file.

    Args:
        records: Sequence of ExperimentRecord objects.
        file_path: Destination JSONL file path.
        append: If True, appends to file; otherwise overwrites.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def load_experiment_records(file_path: Union[str, Path]) -> List[ExperimentRecord]:
    """Load ExperimentRecord objects from a JSONL file.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        List of ExperimentRecord objects.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment results file not found: {file_path}")

    records: List[ExperimentRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                data = json.loads(line_str)
                records.append(ExperimentRecord.from_dict(data))
    return records
