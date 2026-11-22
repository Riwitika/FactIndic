"""Hybrid Claim-Level Natural Language Inference (NLI) Verification.

Integrates atomic claim extraction, BM25 lexical evidence retrieval from source
articles, and multilingual NLI classification (XNLIVerifier) to assess factual
consistency at fine-grained claim granularity.

Methodology & Premise-Hypothesis Formulation:
    - Premise (P): Discrete evidence passage retrieved from the original source article.
    - Hypothesis (H): Extracted atomic factual claim.
    - Verification Strategy: 'hybrid_claim_level'

Epistemic Bounds & Quality Safeguards:
    - Evidence is retrieved EXCLUSIVELY from the original source article.
    - Never classifies a claim as supported merely because BM25 retrieved a passage.
      High lexical overlap does not entail factual truth.
    - Entailment probability reflects consistency between the candidate evidence passage
      and the atomic claim; it does NOT constitute real-world ground truth.
    - Handles multiple evidence passages per claim while preserving retrieval rank ordering.
    - If evidence is missing, zero-score/unrelated, or retrieval fails, an explicit
      structured status ('NO_EVIDENCE' or 'FAILED') is recorded without inventing text.
    - Non-destructive: source GenerationRecord, ExtractedClaim, and ClaimEvidenceResult
      instances remain completely unmodified.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from factindic.generation import GenerationRecord
from factindic.verification.claim_extractor import ExtractedClaim
from factindic.verification.config import NLIConfig
from factindic.verification.evidence_retriever import (
    BM25EvidenceRetriever,
    ClaimEvidenceResult,
    _get_record_id,
)
from factindic.verification.nli import (
    SUPPORTED_VERIFICATION_LANGUAGES,
    XNLIVerifier,
)
from factindic.verification.result import NLILabel, NLIResult


@dataclass
class ClaimVerificationResult:
    """Structured result from hybrid claim-level NLI verification.

    Attributes:
        record_id: Identifier of the originating record (e.g. source_record_id).
        claim_id: Unique identifier of the evaluated atomic claim.
        original_sentence_index: 0-indexed position of originating sentence in summary.
        original_sentence_text: Full text of the originating summary sentence.
        atomic_claim_text: Text of the standalone atomic factual claim (hypothesis).
        evidence_passage: Text of the retrieved evidence passage (premise).
        evidence_start_offset: Start character index in the source article text.
        evidence_end_offset: End character index (exclusive) in the source article text.
        evidence_rank: BM25 retrieval rank of the passage (1 = most relevant).
        evidence_score: BM25 lexical relevance score.
        language: Language code ('en' or 'hi').
        predicted_label: Predicted NLI label ('entailment', 'neutral', 'contradiction', or 'error').
        entailment_prob: Normalized probability of entailment.
        neutral_prob: Normalized probability of neutral/unsupported.
        contradiction_prob: Normalized probability of contradiction.
        probabilities: Full dictionary of class probabilities.
        model_id: Hugging Face model identifier used for NLI evaluation.
        latency_ms: Inference execution time in milliseconds.
        verification_strategy: Strategy identifier ('hybrid_claim_level').
        status: Status code ('SUCCESS', 'NO_EVIDENCE', or 'FAILED').
        error_message: Optional error description if verification or retrieval failed.
        verification_timestamp_utc: Execution timestamp in ISO format.
        prompt_strategy: Prompt condition under which the summary was generated.
        source_dataset: Originating benchmark dataset (e.g. 'csebuetnlp/xlsum').
    """

    record_id: str
    claim_id: str
    original_sentence_index: int
    original_sentence_text: str
    atomic_claim_text: str
    evidence_passage: str
    evidence_start_offset: int
    evidence_end_offset: int
    evidence_rank: int
    evidence_score: float
    language: str
    predicted_label: str
    entailment_prob: float
    neutral_prob: float
    contradiction_prob: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    model_id: str = ""
    latency_ms: float = 0.0
    verification_strategy: str = "hybrid_claim_level"
    status: str = "SUCCESS"
    error_message: Optional[str] = None
    verification_timestamp_utc: str = ""
    prompt_strategy: Optional[str] = None
    source_dataset: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.probabilities and self.status == "SUCCESS":
            self.probabilities = {
                NLILabel.CONTRADICTION.value: self.contradiction_prob,
                NLILabel.NEUTRAL.value: self.neutral_prob,
                NLILabel.ENTAILMENT.value: self.entailment_prob,
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert result object to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimVerificationResult":
        """Construct ClaimVerificationResult from dictionary representation."""
        return cls(
            record_id=str(data.get("record_id", "")),
            claim_id=str(data.get("claim_id", "")),
            original_sentence_index=int(data.get("original_sentence_index", 0)),
            original_sentence_text=str(data.get("original_sentence_text", "")),
            atomic_claim_text=str(data.get("atomic_claim_text", "")),
            evidence_passage=str(data.get("evidence_passage", "")),
            evidence_start_offset=int(data.get("evidence_start_offset", 0)),
            evidence_end_offset=int(data.get("evidence_end_offset", 0)),
            evidence_rank=int(data.get("evidence_rank", 1)),
            evidence_score=float(data.get("evidence_score", 0.0)),
            language=str(data.get("language", "")),
            predicted_label=str(data.get("predicted_label", "")),
            entailment_prob=float(data.get("entailment_prob", 0.0)),
            neutral_prob=float(data.get("neutral_prob", 0.0)),
            contradiction_prob=float(data.get("contradiction_prob", 0.0)),
            probabilities=dict(data.get("probabilities", {})),
            model_id=str(data.get("model_id", "")),
            latency_ms=float(data.get("latency_ms", 0.0)),
            verification_strategy=str(data.get("verification_strategy", "hybrid_claim_level")),
            status=str(data.get("status", "SUCCESS")),
            error_message=data.get("error_message"),
            verification_timestamp_utc=str(data.get("verification_timestamp_utc", "")),
            prompt_strategy=data.get("prompt_strategy"),
            source_dataset=data.get("source_dataset"),
        )


class HybridClaimVerifier:
    """Verifies extracted atomic claims against local source article evidence passages.

    Combines BM25 lexical evidence retrieval with multilingual NLI classification.
    """

    def __init__(
        self,
        verifier: Optional[XNLIVerifier] = None,
        retriever: Optional[BM25EvidenceRetriever] = None,
        config: Optional[NLIConfig] = None,
        top_k_evidence: int = 3,
        min_retrieval_score: float = 0.0,
    ) -> None:
        """Initialize the hybrid claim verifier.

        Args:
            verifier: Optional pre-configured XNLIVerifier instance.
            retriever: Optional pre-configured BM25EvidenceRetriever instance.
            config: Optional NLIConfig to construct verifier if not provided.
            top_k_evidence: Maximum number of ranked evidence passages per claim.
            min_retrieval_score: Minimum BM25 score threshold to qualify as candidate evidence.
        """
        self.verifier = verifier or XNLIVerifier(config=config)
        self.retriever = retriever or BM25EvidenceRetriever(min_score=min_retrieval_score)
        self.top_k_evidence = max(1, top_k_evidence)
        self.min_retrieval_score = min_retrieval_score

    def verify_claim(
        self,
        claim: ExtractedClaim,
        source_text: str,
        evidence_results: Optional[Sequence[ClaimEvidenceResult]] = None,
    ) -> List[ClaimVerificationResult]:
        """Verify a single ExtractedClaim against its source article evidence passages.

        Args:
            claim: ExtractedClaim instance.
            source_text: Complete text of the source article.
            evidence_results: Optional pre-computed ClaimEvidenceResult list for this claim.

        Returns:
            List of ClaimVerificationResult objects, one per evidence passage evaluated.
        """
        return self._verify_claims_batch(
            claims=[claim],
            source_text_map={getattr(claim, "record_id", ""): source_text},
            precomputed_evidence_map=(
                {getattr(claim, "claim_id", ""): list(evidence_results)}
                if evidence_results is not None
                else None
            ),
        )

    def verify_record(
        self,
        record: Any,
        claims: Sequence[ExtractedClaim],
        evidence_results: Optional[Sequence[ClaimEvidenceResult]] = None,
    ) -> List[ClaimVerificationResult]:
        """Verify all claims associated with a single record.

        Args:
            record: GenerationRecord or FactIndicRecord providing source_text and metadata.
            claims: Sequence of ExtractedClaim objects for this record.
            evidence_results: Optional pre-computed evidence results for these claims.

        Returns:
            List of ClaimVerificationResult objects preserving claim and evidence order.
        """
        rec_id = _get_record_id(record)
        source_text = getattr(record, "source_text", "")
        record_claims = [c for c in claims if not rec_id or getattr(c, "record_id", "") == rec_id]

        precomputed_map = None
        if evidence_results is not None:
            precomputed_map = {}
            for ev in evidence_results:
                precomputed_map.setdefault(ev.claim_id, []).append(ev)

        return self._verify_claims_batch(
            claims=record_claims,
            source_text_map={rec_id: source_text},
            precomputed_evidence_map=precomputed_map,
            record_metadata_map={rec_id: record},
        )

    def verify_records(
        self,
        records: Sequence[Any],
        claims: Sequence[ExtractedClaim],
        evidence_results: Optional[Sequence[ClaimEvidenceResult]] = None,
        batch_size: Optional[int] = None,
    ) -> List[ClaimVerificationResult]:
        """Verify claims across multiple records using batched NLI inference.

        Maintains model in memory across records without reloading.

        Args:
            records: Sequence of GenerationRecord or FactIndicRecord objects.
            claims: Sequence of ExtractedClaim objects across all records.
            evidence_results: Optional pre-computed evidence results for these claims.
            batch_size: Optional batch size for batched NLI evaluation.

        Returns:
            List of ClaimVerificationResult objects preserving record, claim, and evidence order.
        """
        source_text_map = {_get_record_id(r, idx): getattr(r, "source_text", "") for idx, r in enumerate(records)}
        record_metadata_map = {_get_record_id(r, idx): r for idx, r in enumerate(records)}

        precomputed_map = None
        if evidence_results is not None:
            precomputed_map = {}
            for ev in evidence_results:
                precomputed_map.setdefault(ev.claim_id, []).append(ev)

        return self._verify_claims_batch(
            claims=claims,
            source_text_map=source_text_map,
            precomputed_evidence_map=precomputed_map,
            record_metadata_map=record_metadata_map,
            batch_size=batch_size,
        )

    def _verify_claims_batch(
        self,
        claims: Sequence[ExtractedClaim],
        source_text_map: Dict[str, str],
        precomputed_evidence_map: Optional[Dict[str, List[ClaimEvidenceResult]]] = None,
        record_metadata_map: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
    ) -> List[ClaimVerificationResult]:
        """Internal execution method running retrieval and batched NLI inference."""
        if not claims:
            return []

        timestamp_utc = datetime.now(timezone.utc).isoformat()
        all_results: List[ClaimVerificationResult] = []
        valid_pairs_to_verify: List[Dict[str, Any]] = []

        for claim in claims:
            rec_id = getattr(claim, "record_id", "")
            claim_id = getattr(claim, "claim_id", "")
            sent_idx = getattr(claim, "original_sentence_index", 0)
            sent_text = getattr(claim, "original_sentence_text", "")
            claim_text = getattr(claim, "atomic_claim_text", "")
            lang = getattr(claim, "language", "")
            prompt_strat = getattr(claim, "prompt_strategy", None)
            source_ds = getattr(claim, "source_dataset", None)

            # Metadata fallback from record
            if record_metadata_map and rec_id in record_metadata_map:
                rec_obj = record_metadata_map[rec_id]
                prompt_strat = prompt_strat or getattr(rec_obj, "prompt_strategy", None)
                source_ds = source_ds or getattr(rec_obj, "source_dataset", None)

            # Check supported language
            lang_clean = str(lang).strip().lower()
            if lang_clean not in SUPPORTED_VERIFICATION_LANGUAGES:
                all_results.append(
                    ClaimVerificationResult(
                        record_id=rec_id,
                        claim_id=claim_id,
                        original_sentence_index=sent_idx,
                        original_sentence_text=sent_text,
                        atomic_claim_text=claim_text,
                        evidence_passage="",
                        evidence_start_offset=0,
                        evidence_end_offset=0,
                        evidence_rank=1,
                        evidence_score=0.0,
                        language=lang,
                        predicted_label="error",
                        entailment_prob=0.0,
                        neutral_prob=0.0,
                        contradiction_prob=0.0,
                        probabilities={},
                        model_id=self.verifier.config.model_id,
                        latency_ms=0.0,
                        verification_strategy="hybrid_claim_level",
                        status="FAILED",
                        error_message=f"Unsupported language '{lang}'. Expected one of {sorted(SUPPORTED_VERIFICATION_LANGUAGES)}.",
                        verification_timestamp_utc=timestamp_utc,
                        prompt_strategy=prompt_strat,
                        source_dataset=source_ds,
                    )
                )
                continue

            # Check empty claim text
            if not claim_text or not isinstance(claim_text, str) or not claim_text.strip():
                all_results.append(
                    ClaimVerificationResult(
                        record_id=rec_id,
                        claim_id=claim_id,
                        original_sentence_index=sent_idx,
                        original_sentence_text=sent_text,
                        atomic_claim_text="",
                        evidence_passage="",
                        evidence_start_offset=0,
                        evidence_end_offset=0,
                        evidence_rank=1,
                        evidence_score=0.0,
                        language=lang,
                        predicted_label="error",
                        entailment_prob=0.0,
                        neutral_prob=0.0,
                        contradiction_prob=0.0,
                        probabilities={},
                        model_id=self.verifier.config.model_id,
                        latency_ms=0.0,
                        verification_strategy="hybrid_claim_level",
                        status="FAILED",
                        error_message=getattr(claim, "error_message", None) or "Atomic claim text is empty or missing.",
                        verification_timestamp_utc=timestamp_utc,
                        prompt_strategy=prompt_strat,
                        source_dataset=source_ds,
                    )
                )
                continue

            # Check if source text is available
            source_text = source_text_map.get(rec_id, "")
            if not source_text or not isinstance(source_text, str) or not source_text.strip():
                all_results.append(
                    ClaimVerificationResult(
                        record_id=rec_id,
                        claim_id=claim_id,
                        original_sentence_index=sent_idx,
                        original_sentence_text=sent_text,
                        atomic_claim_text=claim_text,
                        evidence_passage="",
                        evidence_start_offset=0,
                        evidence_end_offset=0,
                        evidence_rank=1,
                        evidence_score=0.0,
                        language=lang,
                        predicted_label="error",
                        entailment_prob=0.0,
                        neutral_prob=0.0,
                        contradiction_prob=0.0,
                        probabilities={},
                        model_id=self.verifier.config.model_id,
                        latency_ms=0.0,
                        verification_strategy="hybrid_claim_level",
                        status="FAILED",
                        error_message=f"Source text is empty or missing for record '{rec_id}'.",
                        verification_timestamp_utc=timestamp_utc,
                        prompt_strategy=prompt_strat,
                        source_dataset=source_ds,
                    )
                )
                continue

            # Retrieve evidence passages (either from precomputed map or via retriever)
            if precomputed_evidence_map is not None and claim_id in precomputed_evidence_map:
                ev_list = precomputed_evidence_map[claim_id]
            else:
                ev_list = self.retriever.retrieve_for_claim(
                    claim=claim,
                    source_text=source_text,
                    top_k=self.top_k_evidence,
                    min_score=self.min_retrieval_score,
                )

            # Handle no-evidence or failed retrieval
            if not ev_list:
                all_results.append(
                    ClaimVerificationResult(
                        record_id=rec_id,
                        claim_id=claim_id,
                        original_sentence_index=sent_idx,
                        original_sentence_text=sent_text,
                        atomic_claim_text=claim_text,
                        evidence_passage="",
                        evidence_start_offset=0,
                        evidence_end_offset=0,
                        evidence_rank=1,
                        evidence_score=0.0,
                        language=lang,
                        predicted_label="neutral",
                        entailment_prob=0.0,
                        neutral_prob=1.0,
                        contradiction_prob=0.0,
                        probabilities={
                            NLILabel.CONTRADICTION.value: 0.0,
                            NLILabel.NEUTRAL.value: 1.0,
                            NLILabel.ENTAILMENT.value: 0.0,
                        },
                        model_id=self.verifier.config.model_id,
                        latency_ms=0.0,
                        verification_strategy="hybrid_claim_level",
                        status="NO_EVIDENCE",
                        error_message="No evidence passages returned by retriever.",
                        verification_timestamp_utc=timestamp_utc,
                        prompt_strategy=prompt_strat,
                        source_dataset=source_ds,
                    )
                )
                continue

            # If top evidence result is NO_EVIDENCE or FAILED
            if ev_list[0].retrieval_status in ("NO_EVIDENCE", "FAILED"):
                for ev in ev_list:
                    status_code = ev.retrieval_status
                    pred_label = "neutral" if status_code == "NO_EVIDENCE" else "error"
                    neut_prob = 1.0 if status_code == "NO_EVIDENCE" else 0.0
                    all_results.append(
                        ClaimVerificationResult(
                            record_id=rec_id,
                            claim_id=claim_id,
                            original_sentence_index=sent_idx,
                            original_sentence_text=sent_text,
                            atomic_claim_text=claim_text,
                            evidence_passage=ev.evidence_passage,
                            evidence_start_offset=ev.evidence_start_offset,
                            evidence_end_offset=ev.evidence_end_offset,
                            evidence_rank=ev.retrieval_rank,
                            evidence_score=ev.relevance_score,
                            language=lang,
                            predicted_label=pred_label,
                            entailment_prob=0.0,
                            neutral_prob=neut_prob,
                            contradiction_prob=0.0,
                            probabilities={
                                NLILabel.CONTRADICTION.value: 0.0,
                                NLILabel.NEUTRAL.value: neut_prob,
                                NLILabel.ENTAILMENT.value: 0.0,
                            }
                            if status_code == "NO_EVIDENCE"
                            else {},
                            model_id=self.verifier.config.model_id,
                            latency_ms=0.0,
                            verification_strategy="hybrid_claim_level",
                            status=status_code,
                            error_message=ev.error_message,
                            verification_timestamp_utc=timestamp_utc,
                            prompt_strategy=prompt_strat,
                            source_dataset=source_ds,
                        )
                    )
                continue

            # Successful evidence passages: queue for NLI inference
            for ev in ev_list:
                placeholder_idx = len(all_results)
                # Append temporary placeholder
                all_results.append(
                    ClaimVerificationResult(
                        record_id=rec_id,
                        claim_id=claim_id,
                        original_sentence_index=sent_idx,
                        original_sentence_text=sent_text,
                        atomic_claim_text=claim_text,
                        evidence_passage=ev.evidence_passage,
                        evidence_start_offset=ev.evidence_start_offset,
                        evidence_end_offset=ev.evidence_end_offset,
                        evidence_rank=ev.retrieval_rank,
                        evidence_score=ev.relevance_score,
                        language=lang,
                        predicted_label="pending",
                        entailment_prob=0.0,
                        neutral_prob=0.0,
                        contradiction_prob=0.0,
                        model_id=self.verifier.config.model_id,
                        verification_timestamp_utc=timestamp_utc,
                        prompt_strategy=prompt_strat,
                        source_dataset=source_ds,
                    )
                )

                valid_pairs_to_verify.append(
                    {
                        "index": placeholder_idx,
                        "premise": ev.evidence_passage,
                        "hypothesis": claim_text,
                        "language": lang_clean,
                        "record_id": rec_id,
                        "claim_id": claim_id,
                        "original_sentence_index": sent_idx,
                        "original_sentence_text": sent_text,
                        "atomic_claim_text": claim_text,
                        "evidence_passage": ev.evidence_passage,
                        "evidence_start_offset": ev.evidence_start_offset,
                        "evidence_end_offset": ev.evidence_end_offset,
                        "evidence_rank": ev.retrieval_rank,
                        "evidence_score": ev.relevance_score,
                        "prompt_strategy": prompt_strat,
                        "source_dataset": source_ds,
                    }
                )

        # Execute batched NLI inference across all queued evidence-claim pairs
        if valid_pairs_to_verify:
            pairs = [(item["premise"], item["hypothesis"]) for item in valid_pairs_to_verify]
            # Since all pairs can potentially vary in language, we infer per language or unified
            languages = [item["language"] for item in valid_pairs_to_verify]
            primary_lang = languages[0] if len(set(languages)) == 1 else "en"

            try:
                nli_results: List[NLIResult] = self.verifier.verify_batch(
                    pairs=pairs,
                    language=primary_lang,
                    batch_size=batch_size,
                )

                for item, nli_res in zip(valid_pairs_to_verify, nli_results):
                    res_idx = item["index"]
                    all_results[res_idx] = ClaimVerificationResult(
                        record_id=item["record_id"],
                        claim_id=item["claim_id"],
                        original_sentence_index=item["original_sentence_index"],
                        original_sentence_text=item["original_sentence_text"],
                        atomic_claim_text=item["atomic_claim_text"],
                        evidence_passage=item["evidence_passage"],
                        evidence_start_offset=item["evidence_start_offset"],
                        evidence_end_offset=item["evidence_end_offset"],
                        evidence_rank=item["evidence_rank"],
                        evidence_score=item["evidence_score"],
                        language=item["language"],
                        predicted_label=nli_res.predicted_label,
                        entailment_prob=nli_res.entailment_prob,
                        neutral_prob=nli_res.neutral_prob,
                        contradiction_prob=nli_res.contradiction_prob,
                        probabilities=nli_res.probabilities,
                        model_id=nli_res.model_id or self.verifier.config.model_id,
                        latency_ms=nli_res.latency_ms,
                        verification_strategy="hybrid_claim_level",
                        status="SUCCESS",
                        error_message=None,
                        verification_timestamp_utc=timestamp_utc,
                        prompt_strategy=item["prompt_strategy"],
                        source_dataset=item["source_dataset"],
                    )

            except Exception as exc:
                for item in valid_pairs_to_verify:
                    res_idx = item["index"]
                    all_results[res_idx] = ClaimVerificationResult(
                        record_id=item["record_id"],
                        claim_id=item["claim_id"],
                        original_sentence_index=item["original_sentence_index"],
                        original_sentence_text=item["original_sentence_text"],
                        atomic_claim_text=item["atomic_claim_text"],
                        evidence_passage=item["evidence_passage"],
                        evidence_start_offset=item["evidence_start_offset"],
                        evidence_end_offset=item["evidence_end_offset"],
                        evidence_rank=item["evidence_rank"],
                        evidence_score=item["evidence_score"],
                        language=item["language"],
                        predicted_label="error",
                        entailment_prob=0.0,
                        neutral_prob=0.0,
                        contradiction_prob=0.0,
                        probabilities={},
                        model_id=self.verifier.config.model_id,
                        latency_ms=0.0,
                        verification_strategy="hybrid_claim_level",
                        status="FAILED",
                        error_message=f"Inference failure: {type(exc).__name__}: {str(exc)}",
                        verification_timestamp_utc=timestamp_utc,
                        prompt_strategy=item["prompt_strategy"],
                        source_dataset=item["source_dataset"],
                    )

        return all_results


def save_claim_verification_results(
    results: Sequence[ClaimVerificationResult],
    file_path: Union[str, Path],
    append: bool = False,
) -> None:
    """Save ClaimVerificationResult objects to a JSONL file.

    Args:
        results: Sequence of ClaimVerificationResult objects.
        file_path: Destination JSONL file path.
        append: If True, appends to the file; otherwise overwrites.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def load_claim_verification_results(file_path: Union[str, Path]) -> List[ClaimVerificationResult]:
    """Load ClaimVerificationResult objects from a JSONL file.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        List of ClaimVerificationResult objects.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Claim verification results file not found: {file_path}")

    results: List[ClaimVerificationResult] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                data = json.loads(line_str)
                results.append(ClaimVerificationResult.from_dict(data))
    return results


def aggregate_claim_verification_results(
    results: Sequence[Union[ClaimVerificationResult, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Aggregate passage-level verification results into claim-level and summary-level metrics.

    Level 1: Passage to Claim Aggregation (Existential Grounding):
        Each atomic claim is verified against top-k BM25 evidence passages.
        A claim is deemed 'entailment' if any candidate passage entails it (max entailment >= 0.50
        or top entailment passage predicts 'entailment').
        Otherwise, if any passage contradicts it (max contradiction >= 0.50), it is 'contradiction'.
        Otherwise, it is 'neutral' (unsupported / no evidence).

    Level 2: Claim to Summary Aggregation:
        Calculates total claims, counts/ratios for entailment, neutral, contradiction,
        and mean maximum entailment score across claims.
    """
    if not results:
        return {
            "strategy": "hybrid_claim_level",
            "total_claims": 0,
            "entailed_claims": 0,
            "neutral_claims": 0,
            "contradicted_claims": 0,
            "entailment_ratio": 0.0,
            "neutral_ratio": 0.0,
            "contradiction_ratio": 0.0,
            "mean_claim_entailment": 0.0,
            "claims_detail": [],
        }

    claims_order: List[str] = []
    claims_map: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        r_dict = r.to_dict() if hasattr(r, "to_dict") else dict(r)
        cid = str(r_dict.get("claim_id", ""))
        if cid not in claims_map:
            claims_order.append(cid)
            claims_map[cid] = []
        claims_map[cid].append(r_dict)

    claims_detail: List[Dict[str, Any]] = []
    entailed_count = 0
    contradicted_count = 0
    neutral_count = 0
    entail_scores: List[float] = []

    for cid in claims_order:
        passages = claims_map[cid]
        first_p = passages[0]
        claim_text = first_p.get("atomic_claim_text", "")
        sent_idx = first_p.get("original_sentence_index", 0)

        valid_passages = [p for p in passages if p.get("status") in ("SUCCESS", "NO_EVIDENCE")]
        eval_passages = valid_passages if valid_passages else passages

        max_entail = max((float(p.get("entailment_prob", 0.0)) for p in eval_passages), default=0.0)
        max_contra = max((float(p.get("contradiction_prob", 0.0)) for p in eval_passages), default=0.0)
        best_p = max(eval_passages, key=lambda p: float(p.get("entailment_prob", 0.0)), default=first_p)

        best_pred = str(best_p.get("predicted_label", "")).lower()
        if best_pred == "entailment" or max_entail >= 0.50:
            c_label = "entailment"
            entailed_count += 1
        elif any(str(p.get("predicted_label", "")).lower() == "contradiction" for p in eval_passages) or max_contra >= 0.50:
            c_label = "contradiction"
            contradicted_count += 1
        else:
            c_label = "neutral"
            neutral_count += 1

        entail_scores.append(max_entail)
        claims_detail.append({
            "claim_id": cid,
            "sentence_index": sent_idx,
            "atomic_claim_text": claim_text,
            "predicted_label": c_label,
            "max_entailment_prob": round(max_entail, 4),
            "max_contradiction_prob": round(max_contra, 4),
            "best_evidence_rank": int(best_p.get("evidence_rank", 1)),
            "best_evidence_score": round(float(best_p.get("evidence_score", 0.0)), 2),
            "best_evidence_passage": str(best_p.get("evidence_passage", ""))[:120],
            "passage_evaluations_count": len(passages),
        })

    n = len(claims_order)
    return {
        "strategy": "hybrid_claim_level",
        "total_claims": n,
        "entailed_claims": entailed_count,
        "neutral_claims": neutral_count,
        "contradicted_claims": contradicted_count,
        "entailment_ratio": round(entailed_count / n, 4) if n > 0 else 0.0,
        "neutral_ratio": round(neutral_count / n, 4) if n > 0 else 0.0,
        "contradiction_ratio": round(contradicted_count / n, 4) if n > 0 else 0.0,
        "mean_claim_entailment": round(sum(entail_scores) / n, 4) if n > 0 else 0.0,
        "claims_detail": claims_detail,
    }


def aggregate_sentence_verification_results(
    results: Sequence[Union[Any, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Aggregate sentence-level verification results into summary-level metrics."""
    if not results:
        return {
            "strategy": "sentence_level",
            "total_sentences": 0,
            "entailed_sentences": 0,
            "neutral_sentences": 0,
            "contradicted_sentences": 0,
            "entailment_ratio": 0.0,
            "neutral_ratio": 0.0,
            "contradiction_ratio": 0.0,
            "mean_sentence_entailment": 0.0,
            "sentences_detail": [],
        }

    sentences_detail: List[Dict[str, Any]] = []
    entailed_count = 0
    contradicted_count = 0
    neutral_count = 0
    entail_scores: List[float] = []

    for r in results:
        r_dict = r.to_dict() if hasattr(r, "to_dict") else dict(r)
        label = str(r_dict.get("predicted_label", "")).lower()
        entail_prob = float(r_dict.get("entailment_prob", 0.0))
        contra_prob = float(r_dict.get("contradiction_prob", 0.0))
        neut_prob = float(r_dict.get("neutral_prob", 0.0))

        if label == "entailment":
            entailed_count += 1
        elif label == "contradiction":
            contradicted_count += 1
        else:
            neutral_count += 1

        entail_scores.append(entail_prob)
        sentences_detail.append({
            "sentence_index": int(r_dict.get("sentence_index", 0)),
            "sentence_text": str(r_dict.get("sentence_text", "")),
            "predicted_label": label,
            "entailment_prob": round(entail_prob, 4),
            "neutral_prob": round(neut_prob, 4),
            "contradiction_prob": round(contra_prob, 4),
        })

    m = len(results)
    return {
        "strategy": "sentence_level",
        "total_sentences": m,
        "entailed_sentences": entailed_count,
        "neutral_sentences": neutral_count,
        "contradicted_sentences": contradicted_count,
        "entailment_ratio": round(entailed_count / m, 4) if m > 0 else 0.0,
        "neutral_ratio": round(neutral_count / m, 4) if m > 0 else 0.0,
        "contradiction_ratio": round(contradicted_count / m, 4) if m > 0 else 0.0,
        "mean_sentence_entailment": round(sum(entail_scores) / m, 4) if m > 0 else 0.0,
        "sentences_detail": sentences_detail,
    }


def aggregate_document_verification_results(
    results: Sequence[Union[Any, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Extract document-level verification metrics."""
    if not results:
        return {
            "strategy": "whole_document",
            "predicted_label": "error",
            "entailment_prob": 0.0,
            "neutral_prob": 0.0,
            "contradiction_prob": 0.0,
            "status": "FAILED",
        }
    r = results[0]
    r_dict = r.to_dict() if hasattr(r, "to_dict") else dict(r)
    return {
        "strategy": "whole_document",
        "predicted_label": str(r_dict.get("predicted_label", "")).lower(),
        "entailment_prob": round(float(r_dict.get("entailment_prob", 0.0)), 4),
        "neutral_prob": round(float(r_dict.get("neutral_prob", 0.0)), 4),
        "contradiction_prob": round(float(r_dict.get("contradiction_prob", 0.0)), 4),
        "status": str(r_dict.get("status", "SUCCESS")),
    }


def aggregate_verification_results(
    strategy: str,
    results: Sequence[Any],
) -> Dict[str, Any]:
    """Unified aggregator dispatching to the appropriate strategy aggregator."""
    if strategy == "hybrid_claim_level":
        return aggregate_claim_verification_results(results)
    elif strategy == "sentence_level":
        return aggregate_sentence_verification_results(results)
    elif strategy == "whole_document":
        return aggregate_document_verification_results(results)
    return {}
