"""Regression tests for atomic claim extraction and robust JSON parsing.

Validates that:
- Clean structured JSON arrays are parsed correctly.
- Structured objects with a "claims" field are parsed correctly.
- Markdown code fences (```json ... ```) are cleanly stripped.
- Conversational reasoning / preambles before/after JSON are discarded.
- Internal reasoning statements are never stored as extracted claims.
- Malformed, empty, and truncated JSON responses are explicitly rejected.
"""

import pytest
from factindic.models.base import BaseLLMClient, GenerationResult
from factindic.verification.claim_extractor import (
    AtomicClaimExtractor,
    ExtractedClaim,
    _parse_claims_json,
)


class MockLLMWithResponses(BaseLLMClient):
    """Mock LLM returning predefined responses for testing claim extraction."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.model_id = "test-model"
        self.provider = "test-provider"

    def generate(self, prompt: str, temperature: float = 0.0, max_new_tokens: int = 256, **kwargs):
        return GenerationResult(
            text=self.response_text,
            model_id=self.model_id,
            provider=self.provider,
            generation_parameters={"temperature": temperature},
            latency_seconds=0.01,
        )


def test_parse_clean_json_array():
    raw = '["First atomic claim.", "Second atomic claim."]'
    claims = _parse_claims_json(raw)
    assert claims == ["First atomic claim.", "Second atomic claim."]


def test_parse_clean_json_object():
    raw = '{"claims": ["Dan Jones warns about knotweed.", "Knotweed takes a decade to eradicate."]}'
    claims = _parse_claims_json(raw)
    assert claims == ["Dan Jones warns about knotweed.", "Knotweed takes a decade to eradicate."]


def test_parse_markdown_code_fences():
    raw = """Here is the extracted information:
```json
[
  "The university conducted a four-year study.",
  "Herbicide treatment showed promising results."
]
```
Please let me know if you need more."""
    claims = _parse_claims_json(raw)
    assert claims == [
        "The university conducted a four-year study.",
        "Herbicide treatment showed promising results.",
    ]


def test_parse_conversational_preamble_and_object():
    raw = """We need to decompose the sentence into atomic factual claims.
Sentence: "Dan Jones warns that media scare stories hinder knotweed control."
Potential claims:
1. Dan Jones issued a warning.
2. Media scare stories hinder knotweed control.

{"claims": ["Dan Jones issued a warning.", "Media scare stories hinder knotweed control."]}
"""
    claims = _parse_claims_json(raw)
    assert claims == [
        "Dan Jones issued a warning.",
        "Media scare stories hinder knotweed control.",
    ]


def test_filter_out_reasoning_prefixes():
    raw = """{
  "claims": [
    "We need to produce claims that are independent.",
    "Dan Jones is a researcher at Swansea University.",
    "Note: this is a warning.",
    "Giant hogweed causes severe skin burns."
  ]
}"""
    claims = _parse_claims_json(raw)
    assert claims == [
        "Dan Jones is a researcher at Swansea University.",
        "Giant hogweed causes severe skin burns.",
    ]


def test_reject_empty_and_whitespace():
    with pytest.raises(ValueError, match="empty or whitespace"):
        _parse_claims_json("   \n\t  ")


def test_reject_truncated_incomplete_json():
    raw = 'We need to decompose this sentence. Potential claims:\n1. Knotweed is bad.\n{"claims": ["Knotweed is bad", "It takes 10'
    with pytest.raises(ValueError):
        _parse_claims_json(raw)


def test_reject_malformed_json():
    raw = "Here are the claims: None found in this sentence."
    with pytest.raises(ValueError):
        _parse_claims_json(raw)


def test_reject_empty_claims_list():
    raw = '{"claims": []}'
    with pytest.raises(ValueError, match="zero non-empty"):
        _parse_claims_json(raw)


def test_extractor_integration_with_preamble():
    preamble_output = """We need to parse: "Snowdon faces rhododendron infestations."
Thinking: Rhododendron is an invasive plant.
Output:
{"claims": ["Snowdon faces rhododendron infestations."]}
"""
    client = MockLLMWithResponses(preamble_output)
    extractor = AtomicClaimExtractor(client=client)
    claims = extractor.extract_from_sentence(
        sentence="Snowdon faces rhododendron infestations.",
        record_id="rec_001",
        sentence_index=0,
        language="en",
    )
    assert len(claims) == 1
    assert claims[0].extraction_status == "SUCCESS"
    assert claims[0].atomic_claim_text == "Snowdon faces rhododendron infestations."
    assert claims[0].error_message is None


def test_extractor_handles_model_truncation_gracefully():
    truncated_output = "We need to parse the sentence. Potential claims: [\"Incomplete..."
    client = MockLLMWithResponses(truncated_output)
    extractor = AtomicClaimExtractor(client=client)
    claims = extractor.extract_from_sentence(
        sentence="Some source sentence.",
        record_id="rec_002",
        sentence_index=0,
        language="en",
    )
    assert len(claims) == 1
    assert claims[0].extraction_status == "FAILED"
    assert claims[0].atomic_claim_text == ""
    assert "No valid JSON array found" in claims[0].error_message
