"""Unit tests for Atomic Claim Extraction in FactIndic.

Tests run 100% offline using mocked BaseLLMClient outputs without network calls.
"""

from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from factindic.generation import GenerationRecord
from factindic.models.base import BaseLLMClient, GenerationResult
from factindic.verification import (
    AtomicClaimExtractor,
    ExtractedClaim,
    load_extracted_claims,
    save_extracted_claims,
)
from factindic.verification.claim_extractor import _parse_claims_json


def _make_sample_generation_record(
    record_id: str = "xlsum_claim_101",
    language: str = "en",
    source_text: str = "ISRO launched Chandrayaan-3 from Sriharikota on Friday.",
    generated_summary: str = "ISRO launched Chandrayaan-3 on Friday and the rover landed on Wednesday.",
    prompt_strategy: str = "prompt_a_standard",
) -> GenerationRecord:
    """Helper to construct a valid GenerationRecord."""
    return GenerationRecord(
        source_dataset="csebuetnlp/xlsum",
        source_config=language,
        source_split="test",
        source_record_id=record_id,
        language=language,
        source_text=source_text,
        reference_summary="Ground truth reference summary.",
        prompt_strategy=prompt_strategy,
        prompt_template_version="1.0.0",
        model_id="command-a-plus-05-2026",
        provider="cohere",
        generation_parameters={"temperature": 0.0},
        generated_summary=generated_summary,
        generation_timestamp_utc="2026-11-21T18:00:00Z",
        latency_seconds=1.0,
        status="SUCCESS",
        error_message=None,
    )


def _make_mock_client(raw_response_text: str) -> MagicMock:
    """Helper to create a mocked BaseLLMClient returning controlled JSON strings."""
    client = MagicMock(spec=BaseLLMClient)
    client.generate.return_value = GenerationResult(
        text=raw_response_text,
        model_id="command-a-plus-05-2026",
        provider="cohere",
        generation_parameters={"temperature": 0.0},
        latency_seconds=0.45,
    )
    return client


# ---------------------------------------------------------------------------
# Test 1 & 2: JSON parsing helper & markdown fences
# ---------------------------------------------------------------------------

def test_parse_claims_json_standard():
    """Parses standard JSON list of strings."""
    raw = '["Claim 1", "Claim 2", "Claim 3"]'
    parsed = _parse_claims_json(raw)
    assert parsed == ["Claim 1", "Claim 2", "Claim 3"]


def test_parse_claims_json_markdown_code_fences():
    """Strips markdown code blocks (```json ... ```) and extracts list."""
    raw = """Here are the extracted claims:
```json
[
  "The Reserve Bank raised interest rates.",
  "The decision was announced on Wednesday."
]
```
Hope this helps!"""
    parsed = _parse_claims_json(raw)
    assert len(parsed) == 2
    assert parsed[0] == "The Reserve Bank raised interest rates."
    assert parsed[1] == "The decision was announced on Wednesday."


def test_parse_claims_json_malformed():
    """Raises ValueError when input contains no valid JSON list."""
    with pytest.raises(ValueError, match="No valid JSON array found"):
        _parse_claims_json("This is not a JSON list at all.")

    with pytest.raises(ValueError, match="Expected JSON list of strings"):
        _parse_claims_json('{"claim": "single object not list"}')


# ---------------------------------------------------------------------------
# Test 3 & 4: Multiple and Single Claim Extraction (English & Hindi)
# ---------------------------------------------------------------------------

def test_extract_claims_english_multiple():
    """Decomposes compound English sentence into multiple atomic claims with traceability."""
    mock_json = '["ISRO launched Chandrayaan-3 on Friday.", "The rover landed on Wednesday."]'
    client = _make_mock_client(mock_json)
    extractor = AtomicClaimExtractor(client=client)

    rec = _make_sample_generation_record(
        record_id="en_rec_01",
        language="en",
        generated_summary="ISRO launched Chandrayaan-3 on Friday and the rover landed on Wednesday.",
    )

    claims = extractor.extract_from_record(rec)

    assert len(claims) == 2

    # Verify claim 0
    assert claims[0].record_id == "en_rec_01"
    assert claims[0].original_sentence_index == 0
    assert claims[0].original_sentence_text == rec.generated_summary
    assert claims[0].claim_id == "en_rec_01_s0_c0"
    assert claims[0].atomic_claim_text == "ISRO launched Chandrayaan-3 on Friday."
    assert claims[0].language == "en"
    assert claims[0].extraction_status == "SUCCESS"

    # Verify claim 1
    assert claims[1].record_id == "en_rec_01"
    assert claims[1].original_sentence_index == 0
    assert claims[1].original_sentence_text == rec.generated_summary
    assert claims[1].claim_id == "en_rec_01_s0_c1"
    assert claims[1].atomic_claim_text == "The rover landed on Wednesday."
    assert claims[1].extraction_status == "SUCCESS"


def test_extract_claims_english_single():
    """Extracts a single atomic claim from an indivisible sentence."""
    mock_json = '["The market closed higher today."]'
    client = _make_mock_client(mock_json)
    extractor = AtomicClaimExtractor(client=client)

    claims = extractor.extract_from_sentence(
        sentence="The market closed higher today.",
        record_id="single_rec",
        sentence_index=0,
        language="en",
    )

    assert len(claims) == 1
    assert claims[0].claim_id == "single_rec_s0_c0"
    assert claims[0].atomic_claim_text == "The market closed higher today."
    assert claims[0].extraction_status == "SUCCESS"


def test_extract_claims_hindi_multiple():
    """Decomposes compound Hindi Devanagari sentence into native Hindi atomic claims."""
    mock_json = '["इसरो ने शुक्रवार को चंद्रयान-3 का प्रक्षेपण किया।", "रोवर ने बुधवार को सफल लैंडिंग की।"]'
    client = _make_mock_client(mock_json)
    extractor = AtomicClaimExtractor(client=client)

    hindi_sentence = "इसरो ने शुक्रवार को चंद्रयान-3 का प्रक्षेपण किया और रोवर ने बुधवार को सफल लैंडिंग की।"
    claims = extractor.extract_from_sentence(
        sentence=hindi_sentence,
        record_id="hi_rec_01",
        sentence_index=0,
        language="hi",
    )

    assert len(claims) == 2
    assert claims[0].record_id == "hi_rec_01"
    assert claims[0].original_sentence_index == 0
    assert claims[0].original_sentence_text == hindi_sentence
    assert claims[0].claim_id == "hi_rec_01_s0_c0"
    assert claims[0].atomic_claim_text == "इसरो ने शुक्रवार को चंद्रयान-3 का प्रक्षेपण किया।"
    assert claims[0].language == "hi"
    assert claims[0].extraction_status == "SUCCESS"

    assert claims[1].claim_id == "hi_rec_01_s0_c1"
    assert claims[1].atomic_claim_text == "रोवर ने बुधवार को सफल लैंडिंग की।"
    assert claims[1].language == "hi"
    assert claims[1].extraction_status == "SUCCESS"


# ---------------------------------------------------------------------------
# Test 5: Multi-sentence record traceability
# ---------------------------------------------------------------------------

def test_claim_traceability_across_sentences():
    """Preserves strict sentence and claim indices across a multi-sentence summary."""
    client = MagicMock(spec=BaseLLMClient)
    # Return 2 claims for sentence 0, 1 claim for sentence 1
    client.generate.side_effect = [
        GenerationResult(text='["Claim A1", "Claim A2"]', model_id="m", provider="p", generation_parameters={}, latency_seconds=0.1),
        GenerationResult(text='["Claim B1"]', model_id="m", provider="p", generation_parameters={}, latency_seconds=0.1),
    ]

    extractor = AtomicClaimExtractor(client=client)
    rec = _make_sample_generation_record(
        record_id="trace_rec",
        generated_summary="Sentence zero has two facts. Sentence one has one fact.",
    )

    claims = extractor.extract_from_record(rec)

    assert len(claims) == 3

    assert claims[0].original_sentence_index == 0
    assert claims[0].original_sentence_text == "Sentence zero has two facts."
    assert claims[0].claim_id == "trace_rec_s0_c0"
    assert claims[0].atomic_claim_text == "Claim A1"

    assert claims[1].original_sentence_index == 0
    assert claims[1].original_sentence_text == "Sentence zero has two facts."
    assert claims[1].claim_id == "trace_rec_s0_c1"
    assert claims[1].atomic_claim_text == "Claim A2"

    assert claims[2].original_sentence_index == 1
    assert claims[2].original_sentence_text == "Sentence one has one fact."
    assert claims[2].claim_id == "trace_rec_s1_c0"
    assert claims[2].atomic_claim_text == "Claim B1"


# ---------------------------------------------------------------------------
# Test 6: Empty inputs and invalid configurations
# ---------------------------------------------------------------------------

def test_empty_sentence_input():
    """Empty sentence produces FAILED status with clear error."""
    client = _make_mock_client('["Some claim"]')
    extractor = AtomicClaimExtractor(client=client)

    claims = extractor.extract_from_sentence(
        sentence="   ",
        record_id="empty_sent_rec",
        sentence_index=0,
        language="en",
    )

    assert len(claims) == 1
    assert claims[0].extraction_status == "FAILED"
    assert "Sentence is empty" in claims[0].error_message
    assert not client.generate.called


def test_empty_summary_in_record():
    """Record with empty generated summary produces FAILED status without crashing."""
    client = _make_mock_client('["Some claim"]')
    extractor = AtomicClaimExtractor(client=client)

    rec = _make_sample_generation_record(generated_summary="")
    claims = extractor.extract_from_record(rec)

    assert len(claims) == 1
    assert claims[0].extraction_status == "FAILED"
    assert "Generated summary is empty" in claims[0].error_message


def test_unsupported_language():
    """Unsupported language returns FAILED status."""
    client = _make_mock_client('["Some claim"]')
    extractor = AtomicClaimExtractor(client=client)

    claims = extractor.extract_from_sentence(
        sentence="Valid sentence.",
        record_id="unsupp_rec",
        sentence_index=0,
        language="esperanto",
    )

    assert len(claims) == 1
    assert claims[0].extraction_status == "FAILED"
    assert "Unsupported language" in claims[0].error_message


def test_missing_client():
    """Extractor without configured client returns FAILED status."""
    extractor = AtomicClaimExtractor(client=None)

    claims = extractor.extract_from_sentence(
        sentence="Valid sentence.",
        record_id="no_client_rec",
        sentence_index=0,
        language="en",
    )

    assert len(claims) == 1
    assert claims[0].extraction_status == "FAILED"
    assert "No LLM client configured" in claims[0].error_message


# ---------------------------------------------------------------------------
# Test 7: Malformed responses and API failures
# ---------------------------------------------------------------------------

def test_malformed_json_response():
    """Non-JSON response from LLM client produces FAILED status with error details."""
    client = _make_mock_client("I cannot output JSON today sorry.")
    extractor = AtomicClaimExtractor(client=client)

    claims = extractor.extract_from_sentence(
        sentence="Valid sentence.",
        record_id="malformed_rec",
        sentence_index=0,
        language="en",
    )

    assert len(claims) == 1
    assert claims[0].extraction_status == "FAILED"
    assert "Claim extraction failed" in claims[0].error_message


def test_llm_api_failure():
    """Exceptions raised by the LLM client are trapped and recorded as FAILED."""
    client = MagicMock(spec=BaseLLMClient)
    client.generate.side_effect = ConnectionError("Cohere API rate limit exceeded")
    extractor = AtomicClaimExtractor(client=client)

    claims = extractor.extract_from_sentence(
        sentence="Valid sentence.",
        record_id="api_fail_rec",
        sentence_index=0,
        language="en",
    )

    assert len(claims) == 1
    assert claims[0].extraction_status == "FAILED"
    assert "Cohere API rate limit exceeded" in claims[0].error_message


# ---------------------------------------------------------------------------
# Test 8: Source record preservation
# ---------------------------------------------------------------------------

def test_preservation_of_source_records():
    """Input GenerationRecord instances remain completely unmodified after claim extraction."""
    mock_json = '["Claim 1", "Claim 2"]'
    client = _make_mock_client(mock_json)
    extractor = AtomicClaimExtractor(client=client)

    rec = _make_sample_generation_record(record_id="preserve_test")
    orig_dict = rec.to_dict()

    _ = extractor.extract_from_record(rec)

    assert rec.to_dict() == orig_dict
    assert rec.source_text == orig_dict["source_text"]
    assert rec.generated_summary == orig_dict["generated_summary"]


# ---------------------------------------------------------------------------
# Test 9: Serialization & persistence roundtrip
# ---------------------------------------------------------------------------

def test_claim_serialization_roundtrip(tmp_path: Path):
    """ExtractedClaim to_dict/from_dict and JSONL persistence preserve all fields."""
    c1 = ExtractedClaim(
        record_id="rec_01",
        language="en",
        original_sentence_index=0,
        original_sentence_text="First sentence.",
        claim_id="rec_01_s0_c0",
        atomic_claim_text="Atomic claim text one.",
        extraction_status="SUCCESS",
        error_message=None,
        extraction_timestamp_utc="2026-11-21T19:00:00Z",
        prompt_strategy="prompt_a_standard",
        source_dataset="csebuetnlp/xlsum",
    )
    c2 = ExtractedClaim(
        record_id="rec_01",
        language="en",
        original_sentence_index=0,
        original_sentence_text="First sentence.",
        claim_id="rec_01_s0_c1",
        atomic_claim_text="Atomic claim text two.",
        extraction_status="SUCCESS",
        error_message=None,
        extraction_timestamp_utc="2026-11-21T19:00:00Z",
        prompt_strategy="prompt_a_standard",
        source_dataset="csebuetnlp/xlsum",
    )

    out_file = tmp_path / "test_claims.jsonl"
    count = save_extracted_claims([c1, c2], out_file)
    assert count == 2

    loaded = load_extracted_claims(out_file)
    assert len(loaded) == 2

    assert loaded[0].record_id == c1.record_id
    assert loaded[0].claim_id == "rec_01_s0_c0"
    assert loaded[0].atomic_claim_text == "Atomic claim text one."
    assert loaded[0].extraction_status == "SUCCESS"

    assert loaded[1].record_id == c2.record_id
    assert loaded[1].claim_id == "rec_01_s0_c1"
    assert loaded[1].atomic_claim_text == "Atomic claim text two."
    assert loaded[1].extraction_status == "SUCCESS"
