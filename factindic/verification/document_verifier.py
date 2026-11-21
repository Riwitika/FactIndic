"""Whole-Document Natural Language Inference (NLI) Verification.

Evaluates generated summaries against their full source articles using
the XNLIVerifier adapter under a single premise-hypothesis formulation.

Methodology & Interpretation:
    - Premise: Complete source article text.
    - Hypothesis: Complete generated summary text.
    - Verification Strategy: 'whole_document'

Note on Granularity & Factuality:
    Whole-document entailment evaluates whether the source article as an aggregate
    context supports the summary as an aggregate statement. It serves as an empirical
    consistency proxy. A high entailment probability does NOT guarantee that every
    sub-claim or individual sentence in the summary is factually accurate, as minor
    hallucinations or unmentioned details can be overshadowed by broad premise support.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from factindic.generation import GenerationRecord
from factindic.verification.config import NLIConfig
from factindic.verification.nli import (
    SUPPORTED_VERIFICATION_LANGUAGES,
    XNLIVerifier,
)
from factindic.verification.result import NLILabel


@dataclass
class DocumentVerificationResult:
    """Structured result from whole-document NLI verification.

    Attributes:
        record_id: Identifier of the evaluated record (e.g. source_record_id).
        language: Language code ('en' or 'hi').
        predicted_label: Predicted NLI class ('entailment', 'neutral', 'contradiction').
        entailment_prob: Probability that the summary is entailed by the article.
        neutral_prob: Probability that the summary is neutral/unsupported.
        contradiction_prob: Probability that the summary conflicts with the article.
        probabilities: Full dictionary of class probabilities.
        model_id: Identifier of the verification model used.
        latency_ms: Inference execution time in milliseconds.
        verification_strategy: Strategy name ('whole_document').
        status: Status string ('SUCCESS' or 'FAILED').
        error_message: Optional error description if verification failed.
        verification_timestamp_utc: ISO timestamp of verification execution.
        prompt_strategy: Prompt condition under which the summary was generated.
        source_dataset: Originating benchmark dataset (e.g., 'csebuetnlp/xlsum').
    """

    record_id: str
    language: str
    predicted_label: str
    entailment_prob: float
    neutral_prob: float
    contradiction_prob: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    model_id: str = ""
    latency_ms: float = 0.0
    verification_strategy: str = "whole_document"
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
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentVerificationResult":
        """Construct DocumentVerificationResult from a dictionary representation."""
        return cls(
            record_id=str(data.get("record_id", "")),
            language=str(data.get("language", "")),
            predicted_label=str(data.get("predicted_label", "")),
            entailment_prob=float(data.get("entailment_prob", 0.0)),
            neutral_prob=float(data.get("neutral_prob", 0.0)),
            contradiction_prob=float(data.get("contradiction_prob", 0.0)),
            probabilities=dict(data.get("probabilities", {})),
            model_id=str(data.get("model_id", "")),
            latency_ms=float(data.get("latency_ms", 0.0)),
            verification_strategy=str(data.get("verification_strategy", "whole_document")),
            status=str(data.get("status", "SUCCESS")),
            error_message=data.get("error_message"),
            verification_timestamp_utc=str(data.get("verification_timestamp_utc", "")),
            prompt_strategy=data.get("prompt_strategy"),
            source_dataset=data.get("source_dataset"),
        )


class WholeDocumentVerifier:
    """Evaluates factual consistency using complete source articles and generated summaries.

    Reuses the existing XNLIVerifier adapter without duplicating model loading or inference.
    Preserves input GenerationRecord objects unchanged and persists verification outputs separately.
    """

    def __init__(
        self,
        verifier: Optional[XNLIVerifier] = None,
        config: Optional[NLIConfig] = None,
    ) -> None:
        """Initialize WholeDocumentVerifier.

        Args:
            verifier: Pre-instantiated XNLIVerifier adapter. If None, initializes a new one.
            config: Optional NLIConfig used if verifier is not provided.
        """
        self.verifier = verifier or XNLIVerifier(config=config)

    def verify_record(self, record: GenerationRecord) -> DocumentVerificationResult:
        """Verify factual consistency for a single GenerationRecord.

        Args:
            record: GenerationRecord containing source_text and generated_summary.

        Returns:
            DocumentVerificationResult with verification status and probabilities.
        """
        results = self.verify_records([record], batch_size=1)
        return results[0]

    def verify_records(
        self,
        records: Sequence[GenerationRecord],
        batch_size: Optional[int] = None,
    ) -> List[DocumentVerificationResult]:
        """Verify factual consistency for a sequence of GenerationRecord objects.

        Maintains model in memory across records to prevent repeated loading.
        Source GenerationRecord instances are strictly preserved and unmodified.

        Args:
            records: Sequence of GenerationRecord objects.
            batch_size: Optional batch size override for batched NLI inference.

        Returns:
            List of DocumentVerificationResult objects corresponding to each input record.
        """
        if not records:
            return []

        results: List[DocumentVerificationResult] = []
        valid_items_to_verify: List[Dict[str, Any]] = []

        # First pass: validate each record individually without crashing the entire batch
        for idx, rec in enumerate(records):
            timestamp_utc = datetime.now(timezone.utc).isoformat()
            rec_id = getattr(rec, "source_record_id", f"record_{idx}")
            lang = getattr(rec, "language", "")
            prompt_strat = getattr(rec, "prompt_strategy", None)
            source_ds = getattr(rec, "source_dataset", None)

            # Check for non-null record
            if not isinstance(rec, GenerationRecord):
                results.append(
                    DocumentVerificationResult(
                        record_id=str(rec_id),
                        language=str(lang),
                        predicted_label="error",
                        entailment_prob=0.0,
                        neutral_prob=0.0,
                        contradiction_prob=0.0,
                        probabilities={},
                        model_id=self.verifier.config.model_id,
                        latency_ms=0.0,
                        verification_strategy="whole_document",
                        status="FAILED",
                        error_message="Input record must be an instance of GenerationRecord.",
                        verification_timestamp_utc=timestamp_utc,
                        prompt_strategy=prompt_strat,
                        source_dataset=source_ds,
                    )
                )
                continue

            # Check language validity
            lang_clean = str(rec.language).strip().lower()
            if lang_clean not in SUPPORTED_VERIFICATION_LANGUAGES:
                results.append(
                    DocumentVerificationResult(
                        record_id=rec.source_record_id,
                        language=rec.language,
                        predicted_label="error",
                        entailment_prob=0.0,
                        neutral_prob=0.0,
                        contradiction_prob=0.0,
                        probabilities={},
                        model_id=self.verifier.config.model_id,
                        latency_ms=0.0,
                        verification_strategy="whole_document",
                        status="FAILED",
                        error_message=f"Unsupported language '{rec.language}'. Supported: en, hi.",
                        verification_timestamp_utc=timestamp_utc,
                        prompt_strategy=rec.prompt_strategy,
                        source_dataset=rec.source_dataset,
                    )
                )
                continue

            # Check non-empty source text (premise)
            if not rec.source_text or not rec.source_text.strip():
                results.append(
                    DocumentVerificationResult(
                        record_id=rec.source_record_id,
                        language=rec.language,
                        predicted_label="error",
                        entailment_prob=0.0,
                        neutral_prob=0.0,
                        contradiction_prob=0.0,
                        probabilities={},
                        model_id=self.verifier.config.model_id,
                        latency_ms=0.0,
                        verification_strategy="whole_document",
                        status="FAILED",
                        error_message="Source text (premise) is empty or missing.",
                        verification_timestamp_utc=timestamp_utc,
                        prompt_strategy=rec.prompt_strategy,
                        source_dataset=rec.source_dataset,
                    )
                )
                continue

            # Check non-empty generated summary (hypothesis)
            if not rec.generated_summary or not rec.generated_summary.strip():
                results.append(
                    DocumentVerificationResult(
                        record_id=rec.source_record_id,
                        language=rec.language,
                        predicted_label="error",
                        entailment_prob=0.0,
                        neutral_prob=0.0,
                        contradiction_prob=0.0,
                        probabilities={},
                        model_id=self.verifier.config.model_id,
                        latency_ms=0.0,
                        verification_strategy="whole_document",
                        status="FAILED",
                        error_message="Generated summary (hypothesis) is empty or missing.",
                        verification_timestamp_utc=timestamp_utc,
                        prompt_strategy=rec.prompt_strategy,
                        source_dataset=rec.source_dataset,
                    )
                )
                continue

            # Eligible for model inference
            valid_items_to_verify.append({
                "record": rec,
                "target_index": idx,
                "timestamp_utc": timestamp_utc,
            })
            # Insert temporary placeholder in results list
            results.append(None)  # type: ignore

        if not valid_items_to_verify:
            return results

        # Execute batched inference on valid pairs
        pairs = [(item["record"].source_text, item["record"].generated_summary) for item in valid_items_to_verify]
        # Language handling: group by canonical language if mixed, or process per item
        # Since XLSum records are processed in uniform or grouped splits, we handle each language group:
        lang_groups: Dict[str, List[int]] = {}
        for local_idx, item in enumerate(valid_items_to_verify):
            canon_l = SUPPORTED_VERIFICATION_LANGUAGES[item["record"].language.strip().lower()]
            lang_groups.setdefault(canon_l, []).append(local_idx)

        for lang, item_indices in lang_groups.items():
            sub_pairs = [pairs[i] for i in item_indices]
            sub_items = [valid_items_to_verify[i] for i in item_indices]

            try:
                nli_results = self.verifier.verify_batch(
                    pairs=sub_pairs,
                    language=lang,
                    batch_size=batch_size,
                )

                for item, nli_res in zip(sub_items, nli_results):
                    rec = item["record"]
                    target_idx = item["target_index"]
                    results[target_idx] = DocumentVerificationResult(
                        record_id=rec.source_record_id,
                        language=rec.language,
                        predicted_label=nli_res.predicted_label,
                        entailment_prob=nli_res.entailment_prob,
                        neutral_prob=nli_res.neutral_prob,
                        contradiction_prob=nli_res.contradiction_prob,
                        probabilities=nli_res.probabilities,
                        model_id=nli_res.model_id,
                        latency_ms=nli_res.latency_ms,
                        verification_strategy="whole_document",
                        status="SUCCESS",
                        error_message=None,
                        verification_timestamp_utc=item["timestamp_utc"],
                        prompt_strategy=rec.prompt_strategy,
                        source_dataset=rec.source_dataset,
                    )

            except Exception as exc:
                for item in sub_items:
                    rec = item["record"]
                    target_idx = item["target_index"]
                    results[target_idx] = DocumentVerificationResult(
                        record_id=rec.source_record_id,
                        language=rec.language,
                        predicted_label="error",
                        entailment_prob=0.0,
                        neutral_prob=0.0,
                        contradiction_prob=0.0,
                        probabilities={},
                        model_id=self.verifier.config.model_id,
                        latency_ms=0.0,
                        verification_strategy="whole_document",
                        status="FAILED",
                        error_message=f"Inference error during verification: {exc}",
                        verification_timestamp_utc=item["timestamp_utc"],
                        prompt_strategy=rec.prompt_strategy,
                        source_dataset=rec.source_dataset,
                    )

        return results


def save_document_verification_results(
    results: Sequence[DocumentVerificationResult],
    file_path: Union[str, Path],
    append: bool = False,
) -> int:
    """Save document verification results to a UTF-8 JSONL file.

    Args:
        results: Sequence of DocumentVerificationResult objects.
        file_path: Output JSONL file path.
        append: If True, appends to file; if False, overwrites.

    Returns:
        Number of verification results written.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"

    count = 0
    with open(path, mode, encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n")
            count += 1

    return count


def load_document_verification_results(
    file_path: Union[str, Path],
) -> List[DocumentVerificationResult]:
    """Load document verification results from a UTF-8 JSONL file.

    Args:
        file_path: Input JSONL file path.

    Returns:
        List of DocumentVerificationResult objects.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Verification results file not found: {path}")

    results: List[DocumentVerificationResult] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(DocumentVerificationResult.from_dict(json.loads(line)))

    return results
