"""Unit tests for Sentence-Level NLI verification in FactIndic.

Tests run 100% offline using mocked XNLIVerifier outputs.
"""

from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from factindic.generation import GenerationRecord
from factindic.verification import (
    NLIConfig,
    NLIResult,
    SentenceLevelVerifier,
    SentenceVerificationResult,
    load_sentence_verification_results,
    save_sentence_verification_results,
    split_into_sentences,
)


def _make_sample_generation_record(
    record_id: str = "xlsum_en_2001",
    language: str = "en",
    source_text: str = "The Reserve Bank of India raised interest rates by 25 basis points on Wednesday. Inflation was cited as the main reason.",
    generated_summary: str = "India's central bank increased interest rates. Inflation drove the decision.",
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
        generation_parameters={"temperature": 0.3},
        generated_summary=generated_summary,
        generation_timestamp_utc="2026-11-21T18:00:00Z",
        latency_seconds=1.1,
        status="SUCCESS",
        error_message=None,
    )


def _make_mock_verifier(
    label: str = "entailment",
    entail_prob: float = 0.94,
    neut_prob: float = 0.05,
    contra_prob: float = 0.01,
) -> MagicMock:
    """Helper to create a mocked XNLIVerifier returning controlled probabilities."""
    mock_verifier = MagicMock()
    mock_verifier.config = NLIConfig(model_id="joeddav/xlm-roberta-large-xnli")

    def mock_verify_batch(pairs, language="en", batch_size=None):
        results = []
        for p, h in pairs:
            results.append(
                NLIResult(
                    premise=p,
                    hypothesis=h,
                    language=language,
                    predicted_label=label,
                    entailment_prob=entail_prob,
                    neutral_prob=neut_prob,
                    contradiction_prob=contra_prob,
                    probabilities={
                        "entailment": entail_prob,
                        "neutral": neut_prob,
                        "contradiction": contra_prob,
                    },
                    model_id="joeddav/xlm-roberta-large-xnli",
                    latency_ms=28.0,
                )
            )
        return results

    mock_verifier.verify_batch.side_effect = mock_verify_batch
    return mock_verifier


# ---------------------------------------------------------------------------
# Test 1 & 2: Multilingual sentence splitting
# ---------------------------------------------------------------------------

def test_sentence_splitting_english():
    """English text is correctly split by periods, question marks, and exclamation marks."""
    text = "First statement. Did something happen? Yes, indeed! Final thought."
    sentences = split_into_sentences(text, language="en")
    assert len(sentences) == 4
    assert sentences[0] == "First statement."
    assert sentences[1] == "Did something happen?"
    assert sentences[2] == "Yes, indeed!"
    assert sentences[3] == "Final thought."


def test_sentence_splitting_hindi():
    """Hindi text is correctly split by purna viram (।), deergh viram (॥), and standard punctuation."""
    text = "इसरो ने चंद्रयान-3 लॉन्च किया। यह एक ऐतिहासिक क्षण था॥ क्या यह सफल रहा? हाँ, बिल्कुल!"
    sentences = split_into_sentences(text, language="hi")
    assert len(sentences) == 4
    assert sentences[0] == "इसरो ने चंद्रयान-3 लॉन्च किया।"
    assert sentences[1] == "यह एक ऐतिहासिक क्षण था॥"
    assert sentences[2] == "क्या यह सफल रहा?"
    assert sentences[3] == "हाँ, बिल्कुल!"


def test_sentence_splitting_edge_cases():
    """Handles multiple spaces, newlines, empty strings, and text without punctuation."""
    assert split_into_sentences("") == []
    assert split_into_sentences("   \n\t  ") == []

    text_newlines = "Sentence one.\n\nSentence two.\nSentence three."
    sents = split_into_sentences(text_newlines)
    assert len(sents) == 3
    assert sents[0] == "Sentence one."
    assert sents[1] == "Sentence two."
    assert sents[2] == "Sentence three."

    no_punct = "Single sentence without trailing punctuation"
    assert split_into_sentences(no_punct) == ["Single sentence without trailing punctuation"]


# ---------------------------------------------------------------------------
# Test 3 & 4: Sentence-level verification (English & Hindi)
# ---------------------------------------------------------------------------

def test_verify_record_english_sentences():
    """SentenceLevelVerifier splits English summary and verifies each sentence in order."""
    mock_v = _make_mock_verifier(label="entailment", entail_prob=0.91)
    verifier = SentenceLevelVerifier(verifier=mock_v)

    rec = _make_sample_generation_record(
        record_id="rec_en_multi",
        language="en",
        source_text="The Indian team won the first match in Sydney. They played exceptionally well.",
        generated_summary="India won the first match. The team performed well.",
    )

    results = verifier.verify_record(rec)

    assert len(results) == 2
    assert results[0].record_id == "rec_en_multi"
    assert results[0].sentence_index == 0
    assert results[0].sentence_text == "India won the first match."
    assert results[0].predicted_label == "entailment"
    assert results[0].entailment_prob == 0.91
    assert results[0].verification_strategy == "sentence_level"
    assert results[0].status == "SUCCESS"

    assert results[1].record_id == "rec_en_multi"
    assert results[1].sentence_index == 1
    assert results[1].sentence_text == "The team performed well."
    assert results[1].predicted_label == "entailment"
    assert results[1].status == "SUCCESS"


def test_verify_record_hindi_sentences():
    """SentenceLevelVerifier splits Hindi summary by purna viram and verifies each sentence."""
    mock_v = _make_mock_verifier(label="entailment", entail_prob=0.88)
    verifier = SentenceLevelVerifier(verifier=mock_v)

    rec = _make_sample_generation_record(
        record_id="rec_hi_multi",
        language="hi",
        source_text="भारतीय अंतरिक्ष अनुसंधान संगठन (इसरो) ने चंद्रयान-3 का सफल प्रक्षेपण किया।",
        generated_summary="इसरो ने चंद्रयान-3 लॉन्च किया। यह मिशन सफल रहा।",
    )

    results = verifier.verify_record(rec)

    assert len(results) == 2
    assert results[0].record_id == "rec_hi_multi"
    assert results[0].sentence_index == 0
    assert results[0].sentence_text == "इसरो ने चंद्रयान-3 लॉन्च किया।"
    assert results[0].language == "hi"
    assert results[0].status == "SUCCESS"

    assert results[1].record_id == "rec_hi_multi"
    assert results[1].sentence_index == 1
    assert results[1].sentence_text == "यह मिशन सफल रहा।"
    assert results[1].language == "hi"
    assert results[1].status == "SUCCESS"


# ---------------------------------------------------------------------------
# Test 5: All three NLI labels across sentences
# ---------------------------------------------------------------------------

def test_verify_record_labels():
    """Verifier handles alternating entailment, neutral, and contradiction outcomes."""
    mock_v = MagicMock()
    mock_v.config = NLIConfig(model_id="joeddav/xlm-roberta-large-xnli")

    # Sequence of outcomes for 3 sentences
    label_seq = [
        ("entailment", 0.95, 0.04, 0.01),
        ("neutral", 0.10, 0.85, 0.05),
        ("contradiction", 0.02, 0.03, 0.95),
    ]

    def mock_verify_batch(pairs, language="en", batch_size=None):
        out = []
        for idx, (p, h) in enumerate(pairs):
            lbl, ent, neu, con = label_seq[idx % len(label_seq)]
            out.append(
                NLIResult(
                    premise=p,
                    hypothesis=h,
                    language=language,
                    predicted_label=lbl,
                    entailment_prob=ent,
                    neutral_prob=neu,
                    contradiction_prob=con,
                    probabilities={"entailment": ent, "neutral": neu, "contradiction": con},
                    model_id="joeddav/xlm-roberta-large-xnli",
                    latency_ms=25.0,
                )
            )
        return out

    mock_v.verify_batch.side_effect = mock_verify_batch
    verifier = SentenceLevelVerifier(verifier=mock_v)

    rec = _make_sample_generation_record(
        generated_summary="Sentence one. Sentence two. Sentence three."
    )

    results = verifier.verify_record(rec)

    assert len(results) == 3
    assert results[0].predicted_label == "entailment"
    assert results[1].predicted_label == "neutral"
    assert results[2].predicted_label == "contradiction"


# ---------------------------------------------------------------------------
# Test 6: Multiple records batching
# ---------------------------------------------------------------------------

def test_verify_records_multiple():
    """Multiple generation records are segmented and verified in batch preserving order."""
    mock_v = _make_mock_verifier()
    verifier = SentenceLevelVerifier(verifier=mock_v)

    records = [
        _make_sample_generation_record(
            record_id="rec_A",
            generated_summary="Summary A1. Summary A2.",
        ),
        _make_sample_generation_record(
            record_id="rec_B",
            generated_summary="Summary B1. Summary B2. Summary B3.",
        ),
    ]

    results = verifier.verify_records(records, batch_size=2)

    # 2 sentences from rec_A + 3 sentences from rec_B = 5 results total
    assert len(results) == 5

    assert results[0].record_id == "rec_A"
    assert results[0].sentence_index == 0
    assert results[0].sentence_text == "Summary A1."

    assert results[1].record_id == "rec_A"
    assert results[1].sentence_index == 1
    assert results[1].sentence_text == "Summary A2."

    assert results[2].record_id == "rec_B"
    assert results[2].sentence_index == 0
    assert results[2].sentence_text == "Summary B1."

    assert results[3].record_id == "rec_B"
    assert results[3].sentence_index == 1
    assert results[3].sentence_text == "Summary B2."

    assert results[4].record_id == "rec_B"
    assert results[4].sentence_index == 2
    assert results[4].sentence_text == "Summary B3."


# ---------------------------------------------------------------------------
# Test 7: Empty and invalid inputs
# ---------------------------------------------------------------------------

def test_empty_source_text():
    """Record with empty premise produces FAILED status with descriptive error."""
    mock_v = _make_mock_verifier()
    verifier = SentenceLevelVerifier(verifier=mock_v)

    rec = _make_sample_generation_record(source_text="")
    results = verifier.verify_record(rec)

    assert len(results) == 1
    assert results[0].status == "FAILED"
    assert "Source text" in results[0].error_message
    assert results[0].verification_strategy == "sentence_level"


def test_empty_generated_summary():
    """Record with empty hypothesis summary produces FAILED status without crashing."""
    mock_v = _make_mock_verifier()
    verifier = SentenceLevelVerifier(verifier=mock_v)

    rec = _make_sample_generation_record(generated_summary="   ")
    results = verifier.verify_record(rec)

    assert len(results) == 1
    assert results[0].status == "FAILED"
    assert "Generated summary" in results[0].error_message


def test_unsupported_language():
    """Record with unsupported language produces FAILED status."""
    mock_v = _make_mock_verifier()
    verifier = SentenceLevelVerifier(verifier=mock_v)

    rec = _make_sample_generation_record(language="latin")
    results = verifier.verify_record(rec)

    assert len(results) == 1
    assert results[0].status == "FAILED"
    assert "Unsupported language" in results[0].error_message


def test_invalid_record_type():
    """Non-GenerationRecord object produces FAILED status."""
    mock_v = _make_mock_verifier()
    verifier = SentenceLevelVerifier(verifier=mock_v)

    results = verifier.verify_records([12345])  # type: ignore
    assert len(results) == 1
    assert results[0].status == "FAILED"
    assert "GenerationRecord" in results[0].error_message


# ---------------------------------------------------------------------------
# Test 8: Inference error handling
# ---------------------------------------------------------------------------

def test_inference_error_handling():
    """Inference exceptions from adapter are trapped and recorded as FAILED."""
    mock_v = MagicMock()
    mock_v.config = NLIConfig(model_id="joeddav/xlm-roberta-large-xnli")
    mock_v.verify_batch.side_effect = RuntimeError("Tokenizer token limit exceeded")

    verifier = SentenceLevelVerifier(verifier=mock_v)
    rec = _make_sample_generation_record()

    results = verifier.verify_record(rec)

    assert len(results) >= 1
    for r in results:
        assert r.status == "FAILED"
        assert "Tokenizer token limit exceeded" in r.error_message


# ---------------------------------------------------------------------------
# Test 9: Source record preservation
# ---------------------------------------------------------------------------

def test_preservation_of_source_records():
    """Input GenerationRecord instances remain completely unmodified after sentence verification."""
    mock_v = _make_mock_verifier()
    verifier = SentenceLevelVerifier(verifier=mock_v)

    rec = _make_sample_generation_record(record_id="rec_unmodified_sent")
    orig_dict = rec.to_dict()

    _ = verifier.verify_record(rec)

    assert rec.to_dict() == orig_dict
    assert rec.source_text == orig_dict["source_text"]
    assert rec.generated_summary == orig_dict["generated_summary"]


# ---------------------------------------------------------------------------
# Test 10: Serialization & persistence roundtrip
# ---------------------------------------------------------------------------

def test_sentence_verification_serialization_roundtrip(tmp_path: Path):
    """SentenceVerificationResult to_dict/from_dict and JSONL persistence preserve all fields."""
    res1 = SentenceVerificationResult(
        record_id="rec_01",
        sentence_index=0,
        sentence_text="First sentence.",
        language="en",
        predicted_label="entailment",
        entailment_prob=0.92,
        neutral_prob=0.06,
        contradiction_prob=0.02,
        probabilities={"entailment": 0.92, "neutral": 0.06, "contradiction": 0.02},
        model_id="joeddav/xlm-roberta-large-xnli",
        latency_ms=22.3,
        verification_strategy="sentence_level",
        status="SUCCESS",
        error_message=None,
        verification_timestamp_utc="2026-11-21T18:45:00Z",
        prompt_strategy="prompt_a_standard",
        source_dataset="csebuetnlp/xlsum",
    )
    res2 = SentenceVerificationResult(
        record_id="rec_01",
        sentence_index=1,
        sentence_text="Second sentence.",
        language="en",
        predicted_label="neutral",
        entailment_prob=0.15,
        neutral_prob=0.80,
        contradiction_prob=0.05,
        probabilities={"entailment": 0.15, "neutral": 0.80, "contradiction": 0.05},
        model_id="joeddav/xlm-roberta-large-xnli",
        latency_ms=21.8,
        verification_strategy="sentence_level",
        status="SUCCESS",
        error_message=None,
        verification_timestamp_utc="2026-11-21T18:45:01Z",
        prompt_strategy="prompt_a_standard",
        source_dataset="csebuetnlp/xlsum",
    )

    out_file = tmp_path / "test_sentence_verifications.jsonl"
    count = save_sentence_verification_results([res1, res2], out_file)
    assert count == 2

    loaded = load_sentence_verification_results(out_file)
    assert len(loaded) == 2

    assert loaded[0].record_id == res1.record_id
    assert loaded[0].sentence_index == 0
    assert loaded[0].sentence_text == "First sentence."
    assert loaded[0].predicted_label == "entailment"
    assert loaded[0].verification_strategy == "sentence_level"

    assert loaded[1].record_id == res2.record_id
    assert loaded[1].sentence_index == 1
    assert loaded[1].sentence_text == "Second sentence."
    assert loaded[1].predicted_label == "neutral"
