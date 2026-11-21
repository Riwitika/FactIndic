"""Sentence-Level Natural Language Inference (NLI) Verification.

Evaluates generated summaries at sentence granularity against complete source articles
using the XNLIVerifier adapter under an iterative premise-hypothesis formulation.

Methodology & Interpretation:
    - Premise: Complete source article text.
    - Hypothesis: Individual summary sentence.
    - Verification Strategy: 'sentence_level'

Epistemic Bounds & Sub-Claim Granularity:
    Evaluating each summary sentence individually provides higher diagnostic resolution
    than whole-document verification by preventing whole-summary masking. However, a high
    entailment probability for a sentence does NOT guarantee that every atomic sub-claim
    within that sentence is true. Compound sentences with conjunctions or relative clauses
    can still mask minor unmentioned or distorted facts.
"""

import json
import re
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


def split_into_sentences(text: str, language: str = "en") -> List[str]:
    """Split a summary into individual sentences supporting English and Hindi.

    Handles English terminal punctuation (. ! ?) and Hindi Devanagari punctuation
    (purna viram । U+0964, deergh viram ॥ U+0965, ? ! .), preserving punctuation
    and original sentence order.

    Args:
        text: Input summary text string.
        language: Language code ('en', 'hi', 'english', 'hindi').

    Returns:
        Ordered list of non-empty sentence strings.
    """
    if not text or not isinstance(text, str):
        return []

    cleaned = text.strip()
    if not cleaned:
        return []

    # Lookbehind for sentence terminators followed by one or more whitespace characters
    pattern = r"(?<=[।॥!?\.])\s+"
    raw_sentences = re.split(pattern, cleaned)

    sentences: List[str] = []
    for s in raw_sentences:
        s_clean = s.strip()
        if s_clean:
            sentences.append(s_clean)

    # Fallback if no delimiter was found
    if not sentences and cleaned:
        sentences.append(cleaned)

    return sentences


@dataclass
class SentenceVerificationResult:
    """Structured result from sentence-level NLI verification.

    Attributes:
        record_id: Identifier of the originating record (e.g. source_record_id).
        sentence_index: 0-indexed position of the sentence within the summary.
        sentence_text: The individual summary sentence text evaluated as the hypothesis.
        language: Language code ('en' or 'hi').
        predicted_label: Predicted NLI class ('entailment', 'neutral', 'contradiction').
        entailment_prob: Probability that this sentence is entailed by the article.
        neutral_prob: Probability that this sentence is neutral/unsupported.
        contradiction_prob: Probability that this sentence conflicts with the article.
        probabilities: Full dictionary of class probabilities.
        model_id: Identifier of the verification model used.
        latency_ms: Inference execution time in milliseconds.
        verification_strategy: Strategy name ('sentence_level').
        status: Status string ('SUCCESS' or 'FAILED').
        error_message: Optional error description if verification failed.
        verification_timestamp_utc: ISO timestamp of verification execution.
        prompt_strategy: Prompt condition under which the summary was generated.
        source_dataset: Originating benchmark dataset (e.g., 'csebuetnlp/xlsum').
    """

    record_id: str
    sentence_index: int
    sentence_text: str
    language: str
    predicted_label: str
    entailment_prob: float
    neutral_prob: float
    contradiction_prob: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    model_id: str = ""
    latency_ms: float = 0.0
    verification_strategy: str = "sentence_level"
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
    def from_dict(cls, data: Dict[str, Any]) -> "SentenceVerificationResult":
        """Construct SentenceVerificationResult from a dictionary representation."""
        return cls(
            record_id=str(data.get("record_id", "")),
            sentence_index=int(data.get("sentence_index", 0)),
            sentence_text=str(data.get("sentence_text", "")),
            language=str(data.get("language", "")),
            predicted_label=str(data.get("predicted_label", "")),
            entailment_prob=float(data.get("entailment_prob", 0.0)),
            neutral_prob=float(data.get("neutral_prob", 0.0)),
            contradiction_prob=float(data.get("contradiction_prob", 0.0)),
            probabilities=dict(data.get("probabilities", {})),
            model_id=str(data.get("model_id", "")),
            latency_ms=float(data.get("latency_ms", 0.0)),
            verification_strategy=str(data.get("verification_strategy", "sentence_level")),
            status=str(data.get("status", "SUCCESS")),
            error_message=data.get("error_message"),
            verification_timestamp_utc=str(data.get("verification_timestamp_utc", "")),
            prompt_strategy=data.get("prompt_strategy"),
            source_dataset=data.get("source_dataset"),
        )


class SentenceLevelVerifier:
    """Evaluates factual consistency by checking each summary sentence against the full article.

    Reuses the existing XNLIVerifier adapter without duplicating model loading or inference.
    Preserves input GenerationRecord objects unchanged and persists verification outputs separately.
    """

    def __init__(
        self,
        verifier: Optional[XNLIVerifier] = None,
        config: Optional[NLIConfig] = None,
    ) -> None:
        """Initialize SentenceLevelVerifier.

        Args:
            verifier: Pre-instantiated XNLIVerifier adapter. If None, initializes a new one.
            config: Optional NLIConfig used if verifier is not provided.
        """
        self.verifier = verifier or XNLIVerifier(config=config)

    def verify_record(self, record: GenerationRecord) -> List[SentenceVerificationResult]:
        """Verify factual consistency for each sentence in a GenerationRecord.

        Args:
            record: GenerationRecord containing source_text and generated_summary.

        Returns:
            List of SentenceVerificationResult objects, one per summary sentence in order.
        """
        return self.verify_records([record])

    def verify_records(
        self,
        records: Sequence[GenerationRecord],
        batch_size: Optional[int] = None,
    ) -> List[SentenceVerificationResult]:
        """Verify factual consistency for sentences across multiple GenerationRecord objects.

        Maintains model in memory across records to prevent repeated loading.
        Source GenerationRecord instances are strictly preserved and unmodified.

        Args:
            records: Sequence of GenerationRecord objects.
            batch_size: Optional batch size override for batched NLI inference.

        Returns:
            List of SentenceVerificationResult objects preserving record and sentence order.
        """
        if not records:
            return []

        all_results: List[SentenceVerificationResult] = []
        valid_pairs_to_verify: List[Dict[str, Any]] = []

        for rec_idx, rec in enumerate(records):
            timestamp_utc = datetime.now(timezone.utc).isoformat()
            rec_id = getattr(rec, "source_record_id", f"record_{rec_idx}")
            lang = getattr(rec, "language", "")
            prompt_strat = getattr(rec, "prompt_strategy", None)
            source_ds = getattr(rec, "source_dataset", None)

            # Check for valid GenerationRecord instance
            if not isinstance(rec, GenerationRecord):
                all_results.append(
                    SentenceVerificationResult(
                        record_id=str(rec_id),
                        sentence_index=0,
                        sentence_text="",
                        language=str(lang),
                        predicted_label="error",
                        entailment_prob=0.0,
                        neutral_prob=0.0,
                        contradiction_prob=0.0,
                        probabilities={},
                        model_id=self.verifier.config.model_id,
                        latency_ms=0.0,
                        verification_strategy="sentence_level",
                        status="FAILED",
                        error_message="Input record must be an instance of GenerationRecord.",
                        verification_timestamp_utc=timestamp_utc,
                        prompt_strategy=prompt_strat,
                        source_dataset=source_ds,
                    )
                )
                continue

            # Check supported language
            lang_clean = str(rec.language).strip().lower()
            if lang_clean not in SUPPORTED_VERIFICATION_LANGUAGES:
                all_results.append(
                    SentenceVerificationResult(
                        record_id=rec.source_record_id,
                        sentence_index=0,
                        sentence_text="",
                        language=rec.language,
                        predicted_label="error",
                        entailment_prob=0.0,
                        neutral_prob=0.0,
                        contradiction_prob=0.0,
                        probabilities={},
                        model_id=self.verifier.config.model_id,
                        latency_ms=0.0,
                        verification_strategy="sentence_level",
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
                all_results.append(
                    SentenceVerificationResult(
                        record_id=rec.source_record_id,
                        sentence_index=0,
                        sentence_text="",
                        language=rec.language,
                        predicted_label="error",
                        entailment_prob=0.0,
                        neutral_prob=0.0,
                        contradiction_prob=0.0,
                        probabilities={},
                        model_id=self.verifier.config.model_id,
                        latency_ms=0.0,
                        verification_strategy="sentence_level",
                        status="FAILED",
                        error_message="Source text (premise) is empty or missing.",
                        verification_timestamp_utc=timestamp_utc,
                        prompt_strategy=rec.prompt_strategy,
                        source_dataset=rec.source_dataset,
                    )
                )
                continue

            # Check non-empty generated summary (hypothesis container)
            if not rec.generated_summary or not rec.generated_summary.strip():
                all_results.append(
                    SentenceVerificationResult(
                        record_id=rec.source_record_id,
                        sentence_index=0,
                        sentence_text="",
                        language=rec.language,
                        predicted_label="error",
                        entailment_prob=0.0,
                        neutral_prob=0.0,
                        contradiction_prob=0.0,
                        probabilities={},
                        model_id=self.verifier.config.model_id,
                        latency_ms=0.0,
                        verification_strategy="sentence_level",
                        status="FAILED",
                        error_message="Generated summary (hypothesis) is empty or missing.",
                        verification_timestamp_utc=timestamp_utc,
                        prompt_strategy=rec.prompt_strategy,
                        source_dataset=rec.source_dataset,
                    )
                )
                continue

            # Segment summary into individual sentences
            sentences = split_into_sentences(rec.generated_summary, language=rec.language)
            if not sentences:
                all_results.append(
                    SentenceVerificationResult(
                        record_id=rec.source_record_id,
                        sentence_index=0,
                        sentence_text="",
                        language=rec.language,
                        predicted_label="error",
                        entailment_prob=0.0,
                        neutral_prob=0.0,
                        contradiction_prob=0.0,
                        probabilities={},
                        model_id=self.verifier.config.model_id,
                        latency_ms=0.0,
                        verification_strategy="sentence_level",
                        status="FAILED",
                        error_message="Summary segmentation produced zero valid sentences.",
                        verification_timestamp_utc=timestamp_utc,
                        prompt_strategy=rec.prompt_strategy,
                        source_dataset=rec.source_dataset,
                    )
                )
                continue

            # For each sentence, register for verification and reserve slot
            for sent_idx, sentence_str in enumerate(sentences):
                target_pos = len(all_results)
                valid_pairs_to_verify.append({
                    "record": rec,
                    "sentence_index": sent_idx,
                    "sentence_text": sentence_str,
                    "target_position": target_pos,
                    "timestamp_utc": timestamp_utc,
                })
                all_results.append(None)  # type: ignore

        if not valid_pairs_to_verify:
            return all_results

        # Group by language to support homogeneous batching
        lang_groups: Dict[str, List[int]] = {}
        for local_idx, item in enumerate(valid_pairs_to_verify):
            canon_l = SUPPORTED_VERIFICATION_LANGUAGES[item["record"].language.strip().lower()]
            lang_groups.setdefault(canon_l, []).append(local_idx)

        for lang, item_indices in lang_groups.items():
            sub_items = [valid_pairs_to_verify[i] for i in item_indices]
            sub_pairs = [(item["record"].source_text, item["sentence_text"]) for item in sub_items]

            try:
                nli_results = self.verifier.verify_batch(
                    pairs=sub_pairs,
                    language=lang,
                    batch_size=batch_size,
                )

                for item, nli_res in zip(sub_items, nli_results):
                    rec = item["record"]
                    target_pos = item["target_position"]
                    all_results[target_pos] = SentenceVerificationResult(
                        record_id=rec.source_record_id,
                        sentence_index=item["sentence_index"],
                        sentence_text=item["sentence_text"],
                        language=rec.language,
                        predicted_label=nli_res.predicted_label,
                        entailment_prob=nli_res.entailment_prob,
                        neutral_prob=nli_res.neutral_prob,
                        contradiction_prob=nli_res.contradiction_prob,
                        probabilities=nli_res.probabilities,
                        model_id=nli_res.model_id,
                        latency_ms=nli_res.latency_ms,
                        verification_strategy="sentence_level",
                        status="SUCCESS",
                        error_message=None,
                        verification_timestamp_utc=item["timestamp_utc"],
                        prompt_strategy=rec.prompt_strategy,
                        source_dataset=rec.source_dataset,
                    )

            except Exception as exc:
                for item in sub_items:
                    rec = item["record"]
                    target_pos = item["target_position"]
                    all_results[target_pos] = SentenceVerificationResult(
                        record_id=rec.source_record_id,
                        sentence_index=item["sentence_index"],
                        sentence_text=item["sentence_text"],
                        language=rec.language,
                        predicted_label="error",
                        entailment_prob=0.0,
                        neutral_prob=0.0,
                        contradiction_prob=0.0,
                        probabilities={},
                        model_id=self.verifier.config.model_id,
                        latency_ms=0.0,
                        verification_strategy="sentence_level",
                        status="FAILED",
                        error_message=f"Inference error during sentence verification: {exc}",
                        verification_timestamp_utc=item["timestamp_utc"],
                        prompt_strategy=rec.prompt_strategy,
                        source_dataset=rec.source_dataset,
                    )

        return all_results


def save_sentence_verification_results(
    results: Sequence[SentenceVerificationResult],
    file_path: Union[str, Path],
    append: bool = False,
) -> int:
    """Save sentence-level verification results to a UTF-8 JSONL file.

    Args:
        results: Sequence of SentenceVerificationResult objects.
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


def load_sentence_verification_results(
    file_path: Union[str, Path],
) -> List[SentenceVerificationResult]:
    """Load sentence-level verification results from a UTF-8 JSONL file.

    Args:
        file_path: Input JSONL file path.

    Returns:
        List of SentenceVerificationResult objects.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Verification results file not found: {path}")

    results: List[SentenceVerificationResult] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(SentenceVerificationResult.from_dict(json.loads(line)))

    return results
