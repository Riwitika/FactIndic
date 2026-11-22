"""Unit and integration tests for RuleBasedClaimExtractor.

Tests English and Hindi clause decomposition, preamble sanitization, subject
propagation, ExtractedClaim schema conformance, and integration with
AtomicClaimExtractor.
"""

import pytest

from factindic.exceptions import QuotaExhaustedError
from factindic.generation import GenerationRecord
from factindic.verification.claim_extractor import AtomicClaimExtractor, ExtractedClaim
from factindic.verification.rule_claim_extractor import (
    RuleBasedClaimExtractor,
    clean_preamble,
    decompose_english_sentence,
    decompose_hindi_sentence,
    is_meta_statement,
)


# ---------------------------------------------------------------------------
# Test 1: Preamble and Reasoning Trace Sanitization
# ---------------------------------------------------------------------------

def test_clean_preamble_english():
    """English reasoning headers, markdown fences, and bullet traces are stripped."""
    raw = """```
The user wants a concise summary in English.
We need to produce a concise summary.
Key points:
- A pedestrian died in Swansea.
- Police confirmed the incident.
Final output:
A pedestrian has died after being hit by a car in Swansea.
```"""
    cleaned = clean_preamble(raw, language="en")
    assert "The user wants" not in cleaned
    assert "We need to produce" not in cleaned
    assert "Key points:" not in cleaned
    assert "Final output:" not in cleaned
    assert "A pedestrian has died after being hit by a car in Swansea." in cleaned


def test_clean_preamble_hindi_with_english_trace():
    """Leading English reasoning preceding Devanagari text is stripped for Hindi."""
    raw = """The user wants a concise summary in Hindi preserving main information.
Need to capture key points: Azadi march by JUI-F ended.
Let's craft ~150-200 words.

<EOS_TOKEN>पाकिस्तान की प्रमुख विपक्षी पार्टी जमीयत-उल-इस्लाम एफ का आज़ादी मार्च बुधवार को समाप्त हो गया।"""

    cleaned = clean_preamble(raw, language="hi")
    assert "The user wants" not in cleaned
    assert "Let's craft" not in cleaned
    assert "<EOS_TOKEN>" not in cleaned
    assert "पाकिस्तान की प्रमुख विपक्षी पार्टी" in cleaned


# ---------------------------------------------------------------------------
# Test 2: Meta-Statement Identification
# ---------------------------------------------------------------------------

def test_is_meta_statement():
    """Identifies pure prompt echoes and instructions vs. factual content."""
    assert is_meta_statement("The user wants a concise summary in Hindi.") is True
    assert is_meta_statement("We need to produce a concise summary.") is True
    assert is_meta_statement("Output the summary in Hindi.") is True
    assert is_meta_statement("Just summary.") is True
    assert is_meta_statement("Proceed.") is True

    # Factual statements should return False
    assert is_meta_statement("A pedestrian has died after being hit by a car in Swansea.") is False
    assert is_meta_statement("अदालत ने आरोपी को दोषी ठहराया और सजा सुनाई।") is False


# ---------------------------------------------------------------------------
# Test 3: English Clause Decomposition
# ---------------------------------------------------------------------------

def test_decompose_english_trailing_attribution():
    """Decomposes trailing attribution into core claim and attribution claim."""
    sent = "A pedestrian has died after being hit by a car in Swansea, South Wales Police have said."
    claims = decompose_english_sentence(sent)
    assert len(claims) == 2
    assert claims[0] == "A pedestrian has died after being hit by a car in Swansea."
    assert "South Wales Police have said that a pedestrian" in claims[1]


def test_decompose_english_leading_attribution():
    """Decomposes leading attribution 'According to X, Y'."""
    sent = "According to health officials, 11 people have died with Covid-19 in Northern Ireland."
    claims = decompose_english_sentence(sent)
    assert len(claims) == 2
    assert "11 people have died with Covid-19 in Northern Ireland." in claims[0]
    assert "According to health officials" in claims[1]


def test_decompose_english_relative_clause():
    """Decomposes non-restrictive relative clause."""
    sent = "The museum, which was built in 1920, reopened yesterday."
    claims = decompose_english_sentence(sent)
    assert len(claims) == 2
    assert claims[0] == "The museum reopened yesterday."
    assert claims[1] == "The museum was built in 1920."


def test_decompose_english_coordinating_conjunction_with_subject():
    """Decomposes coordinate clauses connected by ', while '."""
    sent = "A further 11 people have died with Covid-19 in Northern Ireland, while 390 new cases were confirmed."
    claims = decompose_english_sentence(sent)
    assert len(claims) == 2
    assert "11 people have died" in claims[0]
    assert "390 new cases were confirmed." in claims[1]


def test_decompose_english_subject_propagation():
    """Propagates governing subject to elliptical coordinate clause."""
    sent = "Alice visited Paris, and also confirmed the booking."
    claims = decompose_english_sentence(sent)
    assert len(claims) == 2
    assert "Alice visited Paris." in claims[0]
    assert "Alice" in claims[1]


# ---------------------------------------------------------------------------
# Test 4: Hindi Clause Decomposition
# ---------------------------------------------------------------------------

def test_decompose_hindi_relative_clause():
    """Decomposes Hindi relative clause (जो ... वह)."""
    sent = "जमीयत-उल-इस्लाम का आज़ादी मार्च, जो 27 अक्टूबर को शुरू हुआ था, बुधवार को समाप्त हो गया।"
    claims = decompose_hindi_sentence(sent)
    assert len(claims) == 2
    assert "बुधवार को समाप्त हो गया।" in claims[0]
    assert "27 अक्टूबर को शुरू हुआ था।" in claims[1]
    assert "जमीयत-उल-इस्लाम का आज़ादी मार्च" in claims[0]
    assert "जमीयत-उल-इस्लाम का आज़ादी मार्च" in claims[1]


def test_decompose_hindi_coordinating_conjunction():
    """Decomposes coordinate Hindi clauses connected by 'और'."""
    sent = "विराट कोहली ने शतक बनाया और मैच जिताया।"
    claims = decompose_hindi_sentence(sent)
    assert len(claims) == 2
    assert "विराट कोहली ने शतक बनाया।" in claims[0]
    assert "विराट कोहली ने" in claims[1]
    assert "मैच जिताया।" in claims[1]


def test_decompose_hindi_subordinating_conjunction():
    """Decomposes Hindi clauses connected by 'हालांकि'."""
    sent = "रक्षा मंत्रालय ने इससे इनकार किया है, हालांकि सैनिकों के प्रतिनिधियों ने इसकी पुष्टि की है।"
    claims = decompose_hindi_sentence(sent)
    assert len(claims) == 2
    assert "रक्षा मंत्रालय ने इससे इनकार किया है।" in claims[0]
    assert "सैनिकों के प्रतिनिधियों ने इसकी पुष्टि की है।" in claims[1]


# ---------------------------------------------------------------------------
# Test 5: RuleBasedClaimExtractor Integration
# ---------------------------------------------------------------------------

def test_extractor_from_sentence_english():
    """Extracts structured ExtractedClaim instances with valid metadata in English."""
    extractor = RuleBasedClaimExtractor()
    sent = "A pedestrian has died after being hit by a car in Swansea, South Wales Police have said."
    claims = extractor.extract_from_sentence(
        sentence=sent,
        record_id="test_rec_01",
        sentence_index=0,
        language="en",
        prompt_strategy="prompt_a",
        source_dataset="bbc",
    )

    assert len(claims) == 2
    assert all(isinstance(c, ExtractedClaim) for c in claims)
    assert all(c.extraction_status == "SUCCESS" for c in claims)
    assert claims[0].claim_id == "test_rec_01_s0_c0"
    assert claims[1].claim_id == "test_rec_01_s0_c1"
    assert claims[0].original_sentence_text == sent
    assert claims[0].language == "en"
    assert claims[0].prompt_strategy == "prompt_a"


def test_extractor_from_sentence_hindi():
    """Extracts structured ExtractedClaim instances with valid metadata in Hindi."""
    extractor = RuleBasedClaimExtractor()
    sent = "अदालत ने आरोपी को दोषी ठहराया और 5 साल की सजा सुनाई।"
    claims = extractor.extract_from_sentence(
        sentence=sent,
        record_id="hi_rec_01",
        sentence_index=1,
        language="hi",
        prompt_strategy="prompt_b",
    )

    assert len(claims) == 2
    assert all(c.extraction_status == "SUCCESS" for c in claims)
    assert claims[0].claim_id == "hi_rec_01_s1_c0"
    assert claims[1].claim_id == "hi_rec_01_s1_c1"
    assert claims[0].language == "hi"


def test_extractor_filters_pure_meta_sentence():
    """Pure meta-commentary sentence produces FAILED claim with clear error message."""
    extractor = RuleBasedClaimExtractor()
    sent = "The user wants a concise summary in Hindi."
    claims = extractor.extract_from_sentence(
        sentence=sent,
        record_id="meta_rec",
        sentence_index=0,
        language="hi",
    )

    assert len(claims) == 1
    assert claims[0].extraction_status == "FAILED"
    assert "meta-commentary" in claims[0].error_message


def test_extractor_from_text():
    """Extracts claims across a full multi-sentence text."""
    extractor = RuleBasedClaimExtractor()
    text = "The museum reopened yesterday. Alice visited Paris, and she met Bob."
    claims = extractor.extract_from_text(
        text=text,
        record_id="multi_rec",
        language="en",
    )

    assert len(claims) >= 3
    assert all(c.extraction_status == "SUCCESS" for c in claims)


def test_extractor_from_record():
    """Extracts claims from a GenerationRecord."""
    extractor = RuleBasedClaimExtractor()
    record = GenerationRecord(
        source_dataset="bbc",
        source_config="default",
        source_split="test",
        source_record_id="rec_gen_01",
        language="en",
        source_text="Source article about a pedestrian incident in Swansea.",
        reference_summary="A pedestrian died in Swansea.",
        prompt_strategy="prompt_a",
        prompt_template_version="v1",
        model_id="command-a-plus-05-2026",
        provider="cohere",
        generation_parameters={},
        generated_summary="A pedestrian has died after being hit by a car in Swansea.",
        generation_timestamp_utc="2026-11-22T00:00:00Z",
        latency_seconds=1.2,
    )

    claims = extractor.extract_from_record(record)
    assert len(claims) == 1
    assert claims[0].record_id == "rec_gen_01"
    assert claims[0].extraction_status == "SUCCESS"
    assert "pedestrian has died" in claims[0].atomic_claim_text


def test_extractor_empty_sentence():
    """Empty or whitespace sentence produces FAILED status."""
    extractor = RuleBasedClaimExtractor()
    claims = extractor.extract_from_sentence(
        sentence="   ",
        record_id="empty_rec",
        sentence_index=0,
        language="en",
    )
    assert len(claims) == 1
    assert claims[0].extraction_status == "FAILED"
    assert "empty or whitespace" in claims[0].error_message


def test_extractor_unsupported_language():
    """Unsupported language returns FAILED status."""
    extractor = RuleBasedClaimExtractor()
    claims = extractor.extract_from_sentence(
        sentence="Bonjour le monde.",
        record_id="fr_rec",
        sentence_index=0,
        language="fr",
    )
    assert len(claims) == 1
    assert claims[0].extraction_status == "FAILED"
    assert "Unsupported language" in claims[0].error_message


# ---------------------------------------------------------------------------
# Test 6: AtomicClaimExtractor Integration with Rule Mode and Fallback
# ---------------------------------------------------------------------------

def test_atomic_claim_extractor_mode_rule():
    """AtomicClaimExtractor configured with mode='rule' delegates to RuleBasedClaimExtractor."""
    extractor = AtomicClaimExtractor(mode="rule")
    sent = "A pedestrian has died after being hit by a car in Swansea, South Wales Police have said."
    claims = extractor.extract_from_sentence(
        sentence=sent,
        record_id="rule_mode_rec",
        sentence_index=0,
        language="en",
    )
    assert len(claims) == 2
    assert all(c.extraction_status == "SUCCESS" for c in claims)


def test_atomic_claim_extractor_fallback_on_quota():
    """AtomicClaimExtractor with fallback_to_rule=True falls back to rule extraction on quota error."""
    class QuotaExhaustedMockClient:
        model = "mock"
        def generate(self, **kwargs):
            raise QuotaExhaustedError("Trial quota 1000 calls exhausted.")

    extractor = AtomicClaimExtractor(client=QuotaExhaustedMockClient(), fallback_to_rule=True)
    sent = "Alice visited Paris, and she met Bob."
    claims = extractor.extract_from_sentence(
        sentence=sent,
        record_id="fallback_rec",
        sentence_index=0,
        language="en",
    )
    assert len(claims) == 2
    assert all(c.extraction_status == "SUCCESS" for c in claims)


# ---------------------------------------------------------------------------
# Test 7: Linguistic Property Preservations
# ---------------------------------------------------------------------------

def test_negation_preservation():
    """Negation markers ('not', 'नहीं') are preserved in extracted claims."""
    en_sent = "The defendant was not found guilty in court."
    en_claims = decompose_english_sentence(en_sent)
    assert any("not found guilty" in c for c in en_claims)

    hi_sent = "मंत्रालय ने इस दावे को स्वीकार नहीं किया।"
    hi_claims = decompose_hindi_sentence(hi_sent)
    assert any("नहीं किया" in c for c in hi_claims)


def test_numbers_and_entities_preserved():
    """Numbers, dates, and named entities are preserved in extracted claims."""
    sent = "On 15 August 1947, India gained independence after 200 years of colonial rule."
    claims = decompose_english_sentence(sent)
    assert "15 August 1947" in claims[0]
    assert "India" in claims[0]
    assert "200 years" in claims[0]


# ---------------------------------------------------------------------------
# Test 8: Noun-Phrase vs. Clausal Coordination Regressions
# ---------------------------------------------------------------------------

def test_hindi_noun_coordination_remains_intact():
    """Coordinated noun lists must NOT be split into incomplete fragment claims."""
    # Example 1: Pant's boundary count
    sent1 = "12 छक्के और 8 चौके शामिल थे।"
    claims1 = decompose_hindi_sentence(sent1)
    assert len(claims1) == 1
    assert "12 छक्के और 8 चौके शामिल थे।" in claims1[0]

    # Example 2: Chemical list
    sent2 = "क्लोरोफ़ॉर्म और मीथेनॉल जैसे हानिकारक रसायन पाए गए।"
    claims2 = decompose_hindi_sentence(sent2)
    assert len(claims2) == 1
    assert "क्लोरोफ़ॉर्म और मीथेनॉल जैसे हानिकारक रसायन पाए गए।" in claims2[0]

    # Example 3: Chemical list inside reporting clause
    sent3 = "विशेषज्ञों का कहना है कि इनमें क्लोरोफ़ॉर्म और मीथेनॉल जैसे हानिकारक रासायन हो सकते हैं।"
    claims3 = decompose_hindi_sentence(sent3)
    assert len(claims3) == 1
    assert "क्लोरोफ़ॉर्म और मीथेनॉल" in claims3[0]

    # Example 4: Prefecture list
    sent4 = "जापान के शिमाने, हिराशिमा और कोचि प्रान्तों के ऊपर से दागेगा।"
    claims4 = decompose_hindi_sentence(sent4)
    assert len(claims4) == 1
    assert "शिमाने, हिराशिमा और कोचि प्रान्तों" in claims4[0]


def test_english_noun_coordination_remains_intact():
    """English noun lists connected by 'and' must NOT be split into fragments."""
    sent1 = "Customs seized two packages containing 19 reptiles, including horned lizards and box turtles."
    claims1 = decompose_english_sentence(sent1)
    assert len(claims1) == 1
    assert "horned lizards and box turtles" in claims1[0]

    sent2 = "The company sells computers and smartphones across Europe."
    claims2 = decompose_english_sentence(sent2)
    assert len(claims2) == 1
    assert "computers and smartphones" in claims2[0]


def test_hindi_clausal_coordination_splits_and_propagates_subject():
    """Independent Hindi clauses connected by coordinating conjunctions are cleanly split."""
    sent = "विराट कोहली ने शतक बनाया और मैच जिताया।"
    claims = decompose_hindi_sentence(sent)
    assert len(claims) == 2
    assert claims[0] == "विराट कोहली ने शतक बनाया।"
    assert claims[1] == "विराट कोहली ने मैच जिताया।"


def test_complex_hindi_sentence_pant_decomposition():
    """Complex Hindi sentence splits on clausal boundary while preserving inner noun coordination."""
    sent = "उन्होंने 38 गेंदों में नाबाद 116 रन बनाए, जिसमें 12 छक्के और 8 चौके शामिल थे (104 रन बाउंड्री से), और उनका स्ट्राइक रेट 305 से अधिक था।"
    claims = decompose_hindi_sentence(sent)
    assert len(claims) == 2
    # Claim 0 must contain the entire unsevered noun list
    assert "12 छक्के और 8 चौके शामिल थे" in claims[0]
    # Claim 1 must contain the second clausal proposition
    assert "उनका स्ट्राइक रेट 305 से अधिक था।" in claims[1]
