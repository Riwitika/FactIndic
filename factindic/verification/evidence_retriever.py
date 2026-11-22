"""Claim-Level Evidence Retrieval from Source Articles.

Implements lightweight, multilingual lexical evidence retrieval (BM25Okapi)
to substantiate extracted atomic claims strictly from their original source articles.

Methodology & Epistemic Boundaries:
    - Evidence is retrieved EXCLUSIVELY from the original source article text.
      No external web search, corpus searching, or LLM-generated evidence is used.
    - Multilingual retrieval supports English and native Hindi Devanagari script
      without translation or paraphrasing.
    - Preserves exact character offsets [start_offset, end_offset] within the source article.
    - BM25 score measures lexical similarity; it does NOT constitute semantic entailment
      or real-world factual truth. If no passage shares lexical overlap or the article is empty,
      a structured NO_EVIDENCE status is recorded.
    - Non-destructive: source GenerationRecord and ExtractedClaim instances are preserved intact.
"""

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from factindic.generation import GenerationRecord
from factindic.verification.claim_extractor import ExtractedClaim
from factindic.verification.nli import SUPPORTED_VERIFICATION_LANGUAGES


@dataclass
class EvidencePassage:
    """A segmented passage (sentence) from a source article with character offsets.

    Attributes:
        passage_index: 0-indexed position of the passage in the segmented article.
        text: Exact string content of the passage.
        start_char: Start character offset in the source article text.
        end_char: End character offset (exclusive) in the source article text.
    """

    passage_index: int
    text: str
    start_char: int
    end_char: int


@dataclass
class ClaimEvidenceResult:
    """Structured result of evidence retrieval for an atomic claim.

    Attributes:
        record_id: Identifier of the originating generation record.
        claim_id: Unique identifier of the extracted atomic claim.
        original_claim_text: Full text of the atomic claim being evaluated.
        evidence_passage: Text of the retrieved evidence passage from the source article.
        evidence_start_offset: Start character index in the source article.
        evidence_end_offset: End character index (exclusive) in the source article.
        retrieval_rank: 1-indexed retrieval ranking (1 = most relevant).
        relevance_score: BM25 lexical similarity score.
        language: Language code ('en' or 'hi').
        retrieval_strategy: Identifier of retrieval strategy ('bm25_lexical').
        retrieval_status: 'SUCCESS', 'NO_EVIDENCE', or 'FAILED'.
        error_message: Optional error message if retrieval encountered a failure.
        retrieval_timestamp_utc: Execution timestamp in ISO format.
    """

    record_id: str
    claim_id: str
    original_claim_text: str
    evidence_passage: str
    evidence_start_offset: int
    evidence_end_offset: int
    retrieval_rank: int
    relevance_score: float
    language: str
    retrieval_strategy: str = "bm25_lexical"
    retrieval_status: str = "SUCCESS"
    error_message: Optional[str] = None
    retrieval_timestamp_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert evidence result to Python dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimEvidenceResult":
        """Construct ClaimEvidenceResult from a dictionary representation."""
        return cls(
            record_id=str(data.get("record_id", "")),
            claim_id=str(data.get("claim_id", "")),
            original_claim_text=str(data.get("original_claim_text", "")),
            evidence_passage=str(data.get("evidence_passage", "")),
            evidence_start_offset=int(data.get("evidence_start_offset", 0)),
            evidence_end_offset=int(data.get("evidence_end_offset", 0)),
            retrieval_rank=int(data.get("retrieval_rank", 1)),
            relevance_score=float(data.get("relevance_score", 0.0)),
            language=str(data.get("language", "")),
            retrieval_strategy=str(data.get("retrieval_strategy", "bm25_lexical")),
            retrieval_status=str(data.get("retrieval_status", "SUCCESS")),
            error_message=data.get("error_message"),
            retrieval_timestamp_utc=str(data.get("retrieval_timestamp_utc", "")),
        )


def tokenize_text(text: str, language: str = "en") -> List[str]:
    """Tokenize English or Hindi text into lowercase lexical tokens.

    Supports Latin characters, digits, and native Devanagari script characters
    (\\u0900-\\u097F) without requiring external tokenizer models.

    Args:
        text: Input text string.
        language: Language code ('en' or 'hi').

    Returns:
        List of lowercase string tokens.
    """
    if not text or not isinstance(text, str):
        return []

    # Match consecutive alphanumeric characters or Devanagari characters
    # \u0900-\u097F covers Devanagari vowels, consonants, matras, virama, and digits
    token_pattern = r"[\u0900-\u097F\w]+"
    tokens = re.findall(token_pattern, text.lower())
    return [t for t in tokens if len(t.strip()) > 0]


def segment_article_into_passages(text: str, language: str = "en") -> List[EvidencePassage]:
    """Segment a source article into discrete sentences/passages with exact character spans.

    Uses sentence delimiters supporting English (. ! ?) and Hindi (। U+0964, ॥ U+0965, ? ! .)
    while recording the exact [start_char, end_char] slice in the original source string.

    Args:
        text: Raw source article string.
        language: Language code ('en', 'hi', 'english', 'hindi').

    Returns:
        Ordered list of EvidencePassage instances.
    """
    if not text or not isinstance(text, str):
        return []

    # Delimiter regex matching sentence-ending punctuation followed by whitespace or string end
    # English: . ! ? ; Hindi: । (U+0964), ॥ (U+0965)
    # We find sentence spans by locating either punctuation boundaries or newlines
    delimiter_pattern = r"(?<=[।॥!?\.])(?:\s+|\n+)|(?:\n\s*\n+)"

    passages: List[EvidencePassage] = []
    current_start = 0

    # Split text while tracking exact positions using regex finditer
    for match in re.finditer(delimiter_pattern, text):
        del_start = match.start()
        del_end = match.end()

        span_text = text[current_start:del_start]
        # Trim leading and trailing whitespace while adjusting start and end offsets
        stripped = span_text.strip()
        if stripped:
            leading_ws = len(span_text) - len(span_text.lstrip())
            actual_start = current_start + leading_ws
            actual_end = actual_start + len(stripped)
            passages.append(
                EvidencePassage(
                    passage_index=len(passages),
                    text=stripped,
                    start_char=actual_start,
                    end_char=actual_end,
                )
            )

        current_start = del_end

    # Handle the trailing passage after the last delimiter
    if current_start < len(text):
        remaining = text[current_start:]
        stripped = remaining.strip()
        if stripped:
            leading_ws = len(remaining) - len(remaining.lstrip())
            actual_start = current_start + leading_ws
            actual_end = actual_start + len(stripped)
            passages.append(
                EvidencePassage(
                    passage_index=len(passages),
                    text=stripped,
                    start_char=actual_start,
                    end_char=actual_end,
                )
            )

    # Fallback if no delimiter was found at all
    if not passages and text.strip():
        stripped = text.strip()
        leading_ws = len(text) - len(text.lstrip())
        passages.append(
            EvidencePassage(
                passage_index=0,
                text=stripped,
                start_char=leading_ws,
                end_char=leading_ws + len(stripped),
            )
        )

    return passages


class BM25Index:
    """Pure-Python BM25Okapi inverted index for lightweight CPU passage ranking.

    Implements BM25Okapi scoring:
        score(D, Q) = sum_{q in Q} IDF(q) * (f(q, D) * (k1 + 1)) / (f(q, D) + k1 * (1 - b + b * (|D| / avgdl)))
        IDF(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))

    Parameters:
        k1: Term frequency saturation parameter (default 1.5).
        b: Document length normalization parameter (default 0.75).
    """

    def __init__(
        self,
        passages: Sequence[EvidencePassage],
        language: str = "en",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.passages = list(passages)
        self.language = language
        self.k1 = k1
        self.b = b
        self.num_docs = len(self.passages)

        # Tokenize passages
        self.doc_tokens: List[List[str]] = [
            tokenize_text(p.text, language=self.language) for p in self.passages
        ]
        self.doc_lens: List[int] = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lens) / self.num_docs if self.num_docs > 0 else 0.0

        # Term frequencies per document and document frequencies
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.doc_freqs: Dict[str, int] = {}

        for tokens in self.doc_tokens:
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.doc_term_freqs.append(tf)

            for term in tf.keys():
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

    def _idf(self, term: str) -> float:
        """Compute Robertson-Spärck Jones IDF with smoothing."""
        n_q = self.doc_freqs.get(term, 0)
        # Smoothed BM25Okapi formula: ln(1 + (N - n + 0.5) / (n + 0.5))
        return math.log(1.0 + (self.num_docs - n_q + 0.5) / (n_q + 0.5))

    def score_query(self, query: str) -> List[Tuple[int, float]]:
        """Score all indexed passages against the query string.

        Args:
            query: Query string (e.g., atomic claim text).

        Returns:
            List of (passage_index, bm25_score) tuples sorted in descending order of score.
        """
        if self.num_docs == 0:
            return []

        query_tokens = tokenize_text(query, language=self.language)
        if not query_tokens:
            return [(idx, 0.0) for idx in range(self.num_docs)]

        scores: List[float] = [0.0] * self.num_docs

        for term in query_tokens:
            if term not in self.doc_freqs:
                continue
            idf = self._idf(term)
            for idx in range(self.num_docs):
                f_qd = self.doc_term_freqs[idx].get(term, 0)
                if f_qd > 0:
                    doc_len = self.doc_lens[idx]
                    denominator = f_qd + self.k1 * (
                        1.0 - self.b + self.b * (doc_len / self.avg_doc_len if self.avg_doc_len > 0 else 1.0)
                    )
                    numerator = f_qd * (self.k1 + 1.0)
                    scores[idx] += idf * (numerator / denominator)

        # Sort descending by score, maintaining original index as tie-breaker
        indexed_scores = [(idx, round(scores[idx], 4)) for idx in range(self.num_docs)]
        indexed_scores.sort(key=lambda item: item[1], reverse=True)
        return indexed_scores


def _get_record_id(record: Any, default_idx: int = 0) -> str:
    """Helper to extract record identifier from GenerationRecord or FactIndicRecord."""
    if hasattr(record, "metadata") and isinstance(getattr(record, "metadata"), dict):
        if "raw_id" in record.metadata and record.metadata["raw_id"]:
            return str(record.metadata["raw_id"])
    if hasattr(record, "source_record_id") and getattr(record, "source_record_id"):
        return str(getattr(record, "source_record_id"))
    if hasattr(record, "example_id") and getattr(record, "example_id"):
        return str(getattr(record, "example_id"))
    if hasattr(record, "id") and getattr(record, "id"):
        return str(getattr(record, "id"))
    return f"record_{default_idx}"


class BM25EvidenceRetriever:
    """Retrieves relevant evidence passages from a source article for extracted atomic claims.

    Operates purely on local CPU using BM25Okapi lexical ranking without external services.
    Enforces strict grounding: evidence is retrieved exclusively from the source article.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, min_score: float = 0.0) -> None:
        """Initialize the BM25 evidence retriever.

        Args:
            k1: BM25 term frequency saturation parameter.
            b: BM25 document length normalization parameter.
            min_score: Minimum BM25 score threshold to consider a passage relevant.
        """
        self.k1 = k1
        self.b = b
        self.min_score = min_score

    def retrieve_for_claim(
        self,
        claim: ExtractedClaim,
        source_text: str,
        top_k: int = 3,
        min_score: Optional[float] = None,
    ) -> List[ClaimEvidenceResult]:
        """Retrieve the top-k most relevant evidence passages for a single atomic claim.

        Args:
            claim: ExtractedClaim instance.
            source_text: Complete text of the source article.
            top_k: Maximum number of passages to retrieve.
            min_score: Optional override for minimum relevance score threshold.

        Returns:
            List of ClaimEvidenceResult objects sorted by rank.
        """
        threshold = self.min_score if min_score is None else min_score
        timestamp = datetime.now(timezone.utc).isoformat()
        lang = getattr(claim, "language", "en")
        claim_id = getattr(claim, "claim_id", "")
        record_id = getattr(claim, "record_id", "")
        claim_text = getattr(claim, "atomic_claim_text", "")

        # Validate language support
        if lang not in SUPPORTED_VERIFICATION_LANGUAGES:
            return [
                ClaimEvidenceResult(
                    record_id=record_id,
                    claim_id=claim_id,
                    original_claim_text=claim_text,
                    evidence_passage="",
                    evidence_start_offset=0,
                    evidence_end_offset=0,
                    retrieval_rank=1,
                    relevance_score=0.0,
                    language=lang,
                    retrieval_status="FAILED",
                    error_message=f"Unsupported language '{lang}'. Expected one of {sorted(SUPPORTED_VERIFICATION_LANGUAGES)}.",
                    retrieval_timestamp_utc=timestamp,
                )
            ]

        # Guard against empty source article
        if not source_text or not isinstance(source_text, str) or not source_text.strip():
            return [
                ClaimEvidenceResult(
                    record_id=record_id,
                    claim_id=claim_id,
                    original_claim_text=claim_text,
                    evidence_passage="",
                    evidence_start_offset=0,
                    evidence_end_offset=0,
                    retrieval_rank=1,
                    relevance_score=0.0,
                    language=lang,
                    retrieval_status="FAILED",
                    error_message="Source text is empty or missing.",
                    retrieval_timestamp_utc=timestamp,
                )
            ]

        # Guard against empty claim text
        if not claim_text or not isinstance(claim_text, str) or not claim_text.strip():
            return [
                ClaimEvidenceResult(
                    record_id=record_id,
                    claim_id=claim_id,
                    original_claim_text="",
                    evidence_passage="",
                    evidence_start_offset=0,
                    evidence_end_offset=0,
                    retrieval_rank=1,
                    relevance_score=0.0,
                    language=lang,
                    retrieval_status="FAILED",
                    error_message="Atomic claim text is empty.",
                    retrieval_timestamp_utc=timestamp,
                )
            ]

        # Segment article into discrete passages with character offsets
        passages = segment_article_into_passages(source_text, language=lang)
        if not passages:
            return [
                ClaimEvidenceResult(
                    record_id=record_id,
                    claim_id=claim_id,
                    original_claim_text=claim_text,
                    evidence_passage="",
                    evidence_start_offset=0,
                    evidence_end_offset=0,
                    retrieval_rank=1,
                    relevance_score=0.0,
                    language=lang,
                    retrieval_status="NO_EVIDENCE",
                    error_message="No passages could be segmented from the source article.",
                    retrieval_timestamp_utc=timestamp,
                )
            ]

        # Index passages and score query
        index = BM25Index(passages, language=lang, k1=self.k1, b=self.b)
        ranked_scores = index.score_query(claim_text)

        # Filter and construct results
        results: List[ClaimEvidenceResult] = []
        valid_matches = [(idx, score) for idx, score in ranked_scores if score > threshold]

        if not valid_matches:
            # No passage had lexical overlap or exceeded threshold
            top_idx, top_score = ranked_scores[0] if ranked_scores else (0, 0.0)
            passage = passages[top_idx]
            return [
                ClaimEvidenceResult(
                    record_id=record_id,
                    claim_id=claim_id,
                    original_claim_text=claim_text,
                    evidence_passage=passage.text if top_score > 0 else "",
                    evidence_start_offset=passage.start_char if top_score > 0 else 0,
                    evidence_end_offset=passage.end_char if top_score > 0 else 0,
                    retrieval_rank=1,
                    relevance_score=top_score,
                    language=lang,
                    retrieval_status="NO_EVIDENCE",
                    error_message="No passage in the source article showed lexical overlap with the claim.",
                    retrieval_timestamp_utc=timestamp,
                )
            ]

        # Take up to top_k matching passages
        selected = valid_matches[: max(1, top_k)]
        for rank, (p_idx, score) in enumerate(selected, start=1):
            passage = passages[p_idx]
            results.append(
                ClaimEvidenceResult(
                    record_id=record_id,
                    claim_id=claim_id,
                    original_claim_text=claim_text,
                    evidence_passage=passage.text,
                    evidence_start_offset=passage.start_char,
                    evidence_end_offset=passage.end_char,
                    retrieval_rank=rank,
                    relevance_score=score,
                    language=lang,
                    retrieval_status="SUCCESS",
                    error_message=None,
                    retrieval_timestamp_utc=timestamp,
                )
            )

        return results

    def retrieve_for_record(
        self,
        record: Any,
        claims: Sequence[ExtractedClaim],
        top_k: int = 3,
        min_score: Optional[float] = None,
    ) -> List[ClaimEvidenceResult]:
        """Retrieve evidence passages for all claims associated with a generation record.

        Args:
            record: GenerationRecord providing source_text and metadata.
            claims: Sequence of ExtractedClaim instances.
            top_k: Maximum number of passages per claim.
            min_score: Optional override for minimum relevance score threshold.

        Returns:
            List of ClaimEvidenceResult objects for all matching claims.
        """
        source_text = getattr(record, "source_text", "")
        record_id = _get_record_id(record)

        # Filter claims matching this record's ID (or process all if record_id is blank)
        record_claims = [c for c in claims if not record_id or getattr(c, "record_id", "") == record_id]

        all_results: List[ClaimEvidenceResult] = []
        for claim in record_claims:
            claim_results = self.retrieve_for_claim(
                claim=claim,
                source_text=source_text,
                top_k=top_k,
                min_score=min_score,
            )
            all_results.extend(claim_results)

        return all_results

    def retrieve_for_records(
        self,
        records: Sequence[Any],
        claims: Sequence[ExtractedClaim],
        top_k: int = 3,
        min_score: Optional[float] = None,
    ) -> List[ClaimEvidenceResult]:
        """Retrieve evidence passages across multiple generation records.

        Maps claims to records by record ID and executes retrieval independently per record.

        Args:
            records: Sequence of GenerationRecord instances.
            claims: Sequence of ExtractedClaim instances.
            top_k: Maximum number of passages per claim.
            min_score: Optional override for minimum relevance score threshold.

        Returns:
            List of all ClaimEvidenceResult objects.
        """
        record_map = {_get_record_id(r, idx): r for idx, r in enumerate(records)}
        results: List[ClaimEvidenceResult] = []

        # Group claims by record_id
        claims_by_record: Dict[str, List[ExtractedClaim]] = {}
        for claim in claims:
            rec_id = getattr(claim, "record_id", "")
            claims_by_record.setdefault(rec_id, []).append(claim)

        for rec_id, rec_claims in claims_by_record.items():
            record = record_map.get(rec_id)
            if record is None:
                # Missing corresponding GenerationRecord
                timestamp = datetime.now(timezone.utc).isoformat()
                for c in rec_claims:
                    results.append(
                        ClaimEvidenceResult(
                            record_id=rec_id,
                            claim_id=getattr(c, "claim_id", ""),
                            original_claim_text=getattr(c, "atomic_claim_text", ""),
                            evidence_passage="",
                            evidence_start_offset=0,
                            evidence_end_offset=0,
                            retrieval_rank=1,
                            relevance_score=0.0,
                            language=getattr(c, "language", "en"),
                            retrieval_status="FAILED",
                            error_message=f"GenerationRecord with id '{rec_id}' not found in provided records.",
                            retrieval_timestamp_utc=timestamp,
                        )
                    )
            else:
                record_results = self.retrieve_for_record(
                    record=record,
                    claims=rec_claims,
                    top_k=top_k,
                    min_score=min_score,
                )
                results.extend(record_results)

        return results


def save_evidence_results(
    results: Sequence[ClaimEvidenceResult],
    file_path: Union[str, Path],
    append: bool = False,
) -> None:
    """Save ClaimEvidenceResult objects to a JSONL file.

    Args:
        results: Sequence of ClaimEvidenceResult objects.
        file_path: Destination JSONL file path.
        append: If True, appends to the file; otherwise overwrites.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def load_evidence_results(file_path: Union[str, Path]) -> List[ClaimEvidenceResult]:
    """Load ClaimEvidenceResult objects from a JSONL file.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        List of ClaimEvidenceResult objects.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Evidence results file not found: {file_path}")

    results: List[ClaimEvidenceResult] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                data = json.loads(line_str)
                results.append(ClaimEvidenceResult.from_dict(data))
    return results
