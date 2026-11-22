"""Atomic Factual Claim Extraction from Generated Summaries.

Decomposes complex summary sentences into standalone, independently verifiable
atomic factual claims while preserving strict traceability to the source sentence
and generation record.

Methodology & Epistemic Boundaries:
    - Extracts atomic claims in their original language (English or Hindi Devanagari)
      without translating or paraphrasing out of context.
    - Preserves exact mappings between every atomic claim and its originating sentence.
    - Does NOT perform NLI verification or evidence retrieval in this phase.
    - Does NOT invent claims, add external world knowledge, or alter original semantics.
"""

import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from factindic.generation import GenerationRecord
from factindic.models.base import BaseLLMClient
from factindic.verification.nli import SUPPORTED_VERIFICATION_LANGUAGES
from factindic.verification.sentence_verifier import split_into_sentences

ENGLISH_CLAIM_EXTRACTION_PROMPT = """You are a rigorous factual claim extraction assistant.
Your task is to decompose the provided summary sentence into one or more standalone, atomic factual claims.

Requirements:
1. Each atomic claim must be an independent, self-contained statement.
2. Resolve pronouns and definite references using only the context in the sentence so each claim makes sense on its own.
3. Do NOT add new information, assumptions, or external facts not present in the sentence.
4. Do NOT change or distort the factual meaning.
5. If the sentence contains only one single indivisible claim, return just that claim.
6. Output MUST be a valid JSON array of strings: ["claim 1", "claim 2", ...]

Sentence:
{sentence}

JSON Response:"""

HINDI_CLAIM_EXTRACTION_PROMPT = """आप एक तथ्य-विश्लेषण सहायक हैं।
आपका कार्य दिए गए सारांश वाक्य को एक या अधिक स्वतंत्र तथ्यात्मक दावों (atomic claims) में विभाजित करना है।

दिशानिर्देश:
1. प्रत्येक दावा स्रोत साक्ष्य के आधार पर स्वतंत्र रूप से सत्यापन योग्य और आत्मनिर्भर (self-contained) होना चाहिए।
2. वाक्य में दिए गए संदर्भ के आधार पर सर्वनामों को स्पष्ट करें ताकि प्रत्येक दावा अपने आप में पूर्ण अर्थ दे।
3. कोई नई जानकारी, बाहरी तथ्य या अनुमान न जोड़ें जो वाक्य में मौजूद न हो।
4. मूल तथ्यात्मक अर्थ को बिल्कुल न बदलें।
5. यदि वाक्य में केवल एक ही अविभाज्य दावा है, तो केवल वही दावा दें।
6. उत्तर केवल एक मान्य JSON सूची (array of strings) के प्रारूप में दें: ["दावा 1", "दावा 2", ...]

वाक्य:
{sentence}

JSON उत्तर:"""


@dataclass
class ExtractedClaim:
    """Structured representation of an extracted atomic factual claim.

    Attributes:
        record_id: Identifier of the originating generation record.
        language: Language code ('en' or 'hi').
        original_sentence_index: 0-indexed position of sentence in the summary.
        original_sentence_text: Exact text of the originating summary sentence.
        claim_id: Unique claim identifier (e.g. '{record_id}_s{sentence_index}_c{claim_index}').
        atomic_claim_text: Text of the standalone atomic factual claim.
        extraction_status: 'SUCCESS' or 'FAILED'.
        error_message: Optional error message if extraction encountered an issue.
        extraction_timestamp_utc: Execution timestamp in ISO format.
        prompt_strategy: Prompt condition under which the summary was generated.
        source_dataset: Originating benchmark dataset name.
    """

    record_id: str
    language: str
    original_sentence_index: int
    original_sentence_text: str
    claim_id: str
    atomic_claim_text: str
    extraction_status: str = "SUCCESS"
    error_message: Optional[str] = None
    extraction_timestamp_utc: str = ""
    prompt_strategy: Optional[str] = None
    source_dataset: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert claim to Python dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedClaim":
        """Construct ExtractedClaim from a dictionary representation."""
        return cls(
            record_id=str(data.get("record_id", "")),
            language=str(data.get("language", "")),
            original_sentence_index=int(data.get("original_sentence_index", 0)),
            original_sentence_text=str(data.get("original_sentence_text", "")),
            claim_id=str(data.get("claim_id", "")),
            atomic_claim_text=str(data.get("atomic_claim_text", "")),
            extraction_status=str(data.get("extraction_status", "SUCCESS")),
            error_message=data.get("error_message"),
            extraction_timestamp_utc=str(data.get("extraction_timestamp_utc", "")),
            prompt_strategy=data.get("prompt_strategy"),
            source_dataset=data.get("source_dataset"),
        )


CLAIMS_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of independently verifiable atomic factual claims.",
        }
    },
    "required": ["claims"],
}

REASONING_PREFIXES = (
    "we need to",
    "the user asks",
    "let's craft",
    "potential claims:",
    "sentence:",
    "note:",
    "draft:",
    "refinement:",
    "final output:",
    "step 1",
    "step 2",
    "according to",
)


def _parse_claims_json(raw_text: str) -> List[str]:
    """Parse JSON list of claims from raw model output, handling markdown code fences,
    conversational preambles, and structured object formats.

    Raises:
        ValueError: If JSON array cannot be extracted or parsed, or contains no valid claims.
    """
    cleaned = raw_text.strip()
    if not cleaned:
        raise ValueError("Raw model output is empty or whitespace.")

    # 1. Check for markdown code fences
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    candidate_texts: List[str] = []
    match = re.search(fence_pattern, cleaned)
    if match:
        candidate_texts.append(match.group(1).strip())
    candidate_texts.append(cleaned)

    parsed_obj: Optional[Any] = None
    for cand in candidate_texts:
        # Direct parsing
        try:
            parsed_obj = json.loads(cand)
            break
        except json.JSONDecodeError:
            pass

        # Try isolating innermost/outermost {...}
        brace_start = cand.find("{")
        brace_end = cand.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                parsed_obj = json.loads(cand[brace_start : brace_end + 1])
                break
            except json.JSONDecodeError:
                pass

        # Try isolating innermost/outermost [...]
        bracket_start = cand.find("[")
        bracket_end = cand.rfind("]")
        if bracket_start != -1 and bracket_end > bracket_start:
            try:
                parsed_obj = json.loads(cand[bracket_start : bracket_end + 1])
                break
            except json.JSONDecodeError:
                pass

    if parsed_obj is None:
        raise ValueError(f"No valid JSON array found in output: {raw_text[:120]}")

    raw_items: List[Any] = []
    if isinstance(parsed_obj, dict):
        if "claims" in parsed_obj and isinstance(parsed_obj["claims"], list):
            raw_items = parsed_obj["claims"]
        else:
            raise ValueError(f"Expected JSON list of strings, but got object keys: {list(parsed_obj.keys())}")
    elif isinstance(parsed_obj, list):
        raw_items = parsed_obj
    else:
        raise ValueError(f"Expected JSON list or object with 'claims', got: {type(parsed_obj).__name__}")

    claims: List[str] = []
    for item in raw_items:
        claim_str = ""
        if isinstance(item, str):
            claim_str = item.strip()
        elif isinstance(item, dict) and "claim" in item:
            claim_str = str(item["claim"]).strip()
        elif isinstance(item, dict) and "text" in item:
            claim_str = str(item["text"]).strip()

        if not claim_str:
            continue

        # Filter out internal reasoning / meta-commentary
        lower_claim = claim_str.lower()
        if any(lower_claim.startswith(prefix) for prefix in REASONING_PREFIXES):
            continue

        claims.append(claim_str)

    if not claims:
        raise ValueError("Parsed JSON contains zero non-empty, non-reasoning claim strings.")

    return claims


class AtomicClaimExtractor:
    """Extracts atomic factual claims from summary sentences using configured LLM client.

    Preserves source GenerationRecord objects unchanged and outputs structured ExtractedClaim
    instances with full sentence-level provenance.
    """

    def __init__(
        self,
        client: Optional[BaseLLMClient] = None,
        temperature: float = 0.0,
        delay_seconds: float = 0.0,
    ) -> None:
        """Initialize AtomicClaimExtractor.

        Args:
            client: Configured BaseLLMClient (e.g. CohereClient with Command A+).
            temperature: Sampling temperature for extraction (default: 0.0 for deterministic output).
            delay_seconds: Optional sleep interval in seconds after API calls.
        """
        self.client = client
        self.temperature = temperature
        self.delay_seconds = delay_seconds

    def build_extraction_prompt(self, sentence: str, language: str) -> str:
        """Build the language-appropriate extraction prompt.

        Args:
            sentence: Individual summary sentence.
            language: Target language ('en' or 'hi').

        Returns:
            Formatted prompt string.
        """
        canon_lang = SUPPORTED_VERIFICATION_LANGUAGES.get(language.strip().lower(), "en")
        if canon_lang == "hi":
            return HINDI_CLAIM_EXTRACTION_PROMPT.format(sentence=sentence)
        return ENGLISH_CLAIM_EXTRACTION_PROMPT.format(sentence=sentence)

    def extract_from_sentence(
        self,
        sentence: str,
        record_id: str,
        sentence_index: int,
        language: str,
        prompt_strategy: Optional[str] = None,
        source_dataset: Optional[str] = None,
    ) -> List[ExtractedClaim]:
        """Extract atomic claims from a single sentence.

        Args:
            sentence: Text of the summary sentence.
            record_id: Identifier of the originating record.
            sentence_index: Position of the sentence in the summary.
            language: Language code ('en' or 'hi').
            prompt_strategy: Optional prompt condition.
            source_dataset: Optional source dataset name.

        Returns:
            List of ExtractedClaim instances.
        """
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        claim_base_id = f"{record_id}_s{sentence_index}"

        # Validate sentence
        if not sentence or not sentence.strip():
            return [
                ExtractedClaim(
                    record_id=record_id,
                    language=language,
                    original_sentence_index=sentence_index,
                    original_sentence_text=sentence,
                    claim_id=f"{claim_base_id}_c0",
                    atomic_claim_text="",
                    extraction_status="FAILED",
                    error_message="Sentence is empty or whitespace.",
                    extraction_timestamp_utc=timestamp_utc,
                    prompt_strategy=prompt_strategy,
                    source_dataset=source_dataset,
                )
            ]

        # Validate language
        lang_clean = str(language).strip().lower()
        if lang_clean not in SUPPORTED_VERIFICATION_LANGUAGES:
            return [
                ExtractedClaim(
                    record_id=record_id,
                    language=language,
                    original_sentence_index=sentence_index,
                    original_sentence_text=sentence,
                    claim_id=f"{claim_base_id}_c0",
                    atomic_claim_text="",
                    extraction_status="FAILED",
                    error_message=f"Unsupported language '{language}'. Supported: en, hi.",
                    extraction_timestamp_utc=timestamp_utc,
                    prompt_strategy=prompt_strategy,
                    source_dataset=source_dataset,
                )
            ]

        if self.client is None:
            return [
                ExtractedClaim(
                    record_id=record_id,
                    language=language,
                    original_sentence_index=sentence_index,
                    original_sentence_text=sentence,
                    claim_id=f"{claim_base_id}_c0",
                    atomic_claim_text="",
                    extraction_status="FAILED",
                    error_message="No LLM client configured for AtomicClaimExtractor.",
                    extraction_timestamp_utc=timestamp_utc,
                    prompt_strategy=prompt_strategy,
                    source_dataset=source_dataset,
                )
            ]

        prompt = self.build_extraction_prompt(sentence=sentence, language=lang_clean)

        try:
            import inspect
            gen_kwargs: Dict[str, Any] = {
                "prompt": prompt,
                "temperature": self.temperature,
                "max_new_tokens": 2048,
            }
            sig = inspect.signature(self.client.generate)
            if "response_format" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                gen_kwargs["response_format"] = {"type": "json_object", "schema": CLAIMS_JSON_SCHEMA}

            gen_result = self.client.generate(**gen_kwargs)
            if self.delay_seconds > 0:
                time.sleep(self.delay_seconds)

            raw_output = gen_result.text
            parsed_claims = _parse_claims_json(raw_output)

            results: List[ExtractedClaim] = []
            for claim_idx, claim_str in enumerate(parsed_claims):
                results.append(
                    ExtractedClaim(
                        record_id=record_id,
                        language=lang_clean,
                        original_sentence_index=sentence_index,
                        original_sentence_text=sentence,
                        claim_id=f"{claim_base_id}_c{claim_idx}",
                        atomic_claim_text=claim_str,
                        extraction_status="SUCCESS",
                        error_message=None,
                        extraction_timestamp_utc=timestamp_utc,
                        prompt_strategy=prompt_strategy,
                        source_dataset=source_dataset,
                    )
                )

            return results

        except Exception as exc:
            return [
                ExtractedClaim(
                    record_id=record_id,
                    language=lang_clean,
                    original_sentence_index=sentence_index,
                    original_sentence_text=sentence,
                    claim_id=f"{claim_base_id}_c0",
                    atomic_claim_text="",
                    extraction_status="FAILED",
                    error_message=f"Claim extraction failed: {exc}",
                    extraction_timestamp_utc=timestamp_utc,
                    prompt_strategy=prompt_strategy,
                    source_dataset=source_dataset,
                )
            ]

    def extract_from_record(self, record: GenerationRecord) -> List[ExtractedClaim]:
        """Extract atomic claims from all sentences of a GenerationRecord.

        Args:
            record: GenerationRecord containing source_text and generated_summary.

        Returns:
            List of ExtractedClaim instances preserving sentence and claim order.
        """
        return self.extract_from_records([record])

    def extract_from_records(
        self,
        records: Sequence[GenerationRecord],
    ) -> List[ExtractedClaim]:
        """Extract atomic claims from a sequence of GenerationRecord objects.

        Preserves source GenerationRecord instances unchanged.

        Args:
            records: Sequence of GenerationRecord objects.

        Returns:
            Ordered list of ExtractedClaim objects for all records and sentences.
        """
        if not records:
            return []

        all_claims: List[ExtractedClaim] = []

        for rec_idx, rec in enumerate(records):
            timestamp_utc = datetime.now(timezone.utc).isoformat()
            rec_id = getattr(rec, "source_record_id", f"record_{rec_idx}")
            lang = getattr(rec, "language", "")
            prompt_strat = getattr(rec, "prompt_strategy", None)
            source_ds = getattr(rec, "source_dataset", None)

            # Check valid GenerationRecord instance
            if not isinstance(rec, GenerationRecord):
                all_claims.append(
                    ExtractedClaim(
                        record_id=str(rec_id),
                        language=str(lang),
                        original_sentence_index=0,
                        original_sentence_text="",
                        claim_id=f"{rec_id}_s0_c0",
                        atomic_claim_text="",
                        extraction_status="FAILED",
                        error_message="Input record must be an instance of GenerationRecord.",
                        extraction_timestamp_utc=timestamp_utc,
                        prompt_strategy=prompt_strat,
                        source_dataset=source_ds,
                    )
                )
                continue

            # Check language
            lang_clean = str(rec.language).strip().lower()
            if lang_clean not in SUPPORTED_VERIFICATION_LANGUAGES:
                all_claims.append(
                    ExtractedClaim(
                        record_id=rec.source_record_id,
                        language=rec.language,
                        original_sentence_index=0,
                        original_sentence_text="",
                        claim_id=f"{rec.source_record_id}_s0_c0",
                        atomic_claim_text="",
                        extraction_status="FAILED",
                        error_message=f"Unsupported language '{rec.language}'. Supported: en, hi.",
                        extraction_timestamp_utc=timestamp_utc,
                        prompt_strategy=rec.prompt_strategy,
                        source_dataset=rec.source_dataset,
                    )
                )
                continue

            # Check generated summary
            if not rec.generated_summary or not rec.generated_summary.strip():
                all_claims.append(
                    ExtractedClaim(
                        record_id=rec.source_record_id,
                        language=rec.language,
                        original_sentence_index=0,
                        original_sentence_text="",
                        claim_id=f"{rec.source_record_id}_s0_c0",
                        atomic_claim_text="",
                        extraction_status="FAILED",
                        error_message="Generated summary is empty or missing.",
                        extraction_timestamp_utc=timestamp_utc,
                        prompt_strategy=rec.prompt_strategy,
                        source_dataset=rec.source_dataset,
                    )
                )
                continue

            # Segment summary into individual sentences
            sentences = split_into_sentences(rec.generated_summary, language=rec.language)
            if not sentences:
                all_claims.append(
                    ExtractedClaim(
                        record_id=rec.source_record_id,
                        language=rec.language,
                        original_sentence_index=0,
                        original_sentence_text="",
                        claim_id=f"{rec.source_record_id}_s0_c0",
                        atomic_claim_text="",
                        extraction_status="FAILED",
                        error_message="Summary segmentation produced zero valid sentences.",
                        extraction_timestamp_utc=timestamp_utc,
                        prompt_strategy=rec.prompt_strategy,
                        source_dataset=rec.source_dataset,
                    )
                )
                continue

            for sent_idx, sentence_str in enumerate(sentences):
                sentence_claims = self.extract_from_sentence(
                    sentence=sentence_str,
                    record_id=rec.source_record_id,
                    sentence_index=sent_idx,
                    language=rec.language,
                    prompt_strategy=rec.prompt_strategy,
                    source_dataset=rec.source_dataset,
                )
                all_claims.extend(sentence_claims)

        return all_claims


def save_extracted_claims(
    claims: Sequence[ExtractedClaim],
    file_path: Union[str, Path],
    append: bool = False,
) -> int:
    """Save extracted atomic claims to a UTF-8 JSONL file.

    Args:
        claims: Sequence of ExtractedClaim objects.
        file_path: Output JSONL file path.
        append: If True, appends to file; if False, overwrites.

    Returns:
        Number of claims written.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"

    count = 0
    with open(path, mode, encoding="utf-8") as f:
        for claim in claims:
            f.write(json.dumps(claim.to_dict(), ensure_ascii=False) + "\n")
            count += 1

    return count


def load_extracted_claims(file_path: Union[str, Path]) -> List[ExtractedClaim]:
    """Load extracted atomic claims from a UTF-8 JSONL file.

    Args:
        file_path: Input JSONL file path.

    Returns:
        List of ExtractedClaim objects.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Claims file not found: {path}")

    claims: List[ExtractedClaim] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            claims.append(ExtractedClaim.from_dict(json.loads(line)))

    return claims
