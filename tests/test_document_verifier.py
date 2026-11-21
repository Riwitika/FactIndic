"""Unit tests for Whole-Document NLI verification in FactIndic.

Tests run 100% offline using mocked XNLIVerifier outputs.
"""

from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from factindic.generation import GenerationRecord
from factindic.verification import (
    DocumentVerificationResult,
    NLIConfig,
    NLIResult,
    WholeDocumentVerifier,
    load_document_verification_results,
    save_document_verification_results,
)


def _make_sample_generation_record(
    record_id: str = "xlsum_en_1001",
    language: str = "en",
    source_text: str = "The Reserve Bank of India raised interest rates by 25 basis points on Wednesday.",
    generated_summary: str = "India's central bank increased interest rates.",
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
        reference_summary="Ground truth summary.",
        prompt_strategy=prompt_strategy,
        prompt_template_version="1.0.0",
        model_id="command-a-plus-05-2026",
        provider="cohere",
        generation_parameters={"temperature": 0.3},
        generated_summary=generated_summary,
        generation_timestamp_utc="2026-11-21T18:00:00Z",
        latency_seconds=1.2,
        status="SUCCESS",
        error_message=None,
    )


def _make_mock_verifier(
    label: str = "entailment",
    entail_prob: float = 0.95,
    neut_prob: float = 0.04,
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
                    latency_ms=35.0,
                )
            )
        return results

    mock_verifier.verify_batch.side_effect = mock_verify_batch
    return mock_verifier


# ---------------------------------------------------------------------------
# Test 1 & 2: English and Hindi record verification
# ---------------------------------------------------------------------------

def test_verify_record_english_success():
    """Whole-document verification succeeds for a standard English record."""
    mock_v = _make_mock_verifier(label="entailment", entail_prob=0.92)
    doc_verifier = WholeDocumentVerifier(verifier=mock_v)

    rec = _make_sample_generation_record(
        record_id="en_test_01",
        language="en",
        source_text="The Indian Space Research Organisation launched Chandrayaan-3 successfully.",
        generated_summary="Chandrayaan-3 was launched by ISRO.",
    )

    result = doc_verifier.verify_record(rec)

    assert result.status == "SUCCESS"
    assert result.record_id == "en_test_01"
    assert result.language == "en"
    assert result.predicted_label == "entailment"
    assert result.entailment_prob == 0.92
    assert result.verification_strategy == "whole_document"
    assert result.model_id == "joeddav/xlm-roberta-large-xnli"
    assert result.error_message is None
    assert result.latency_ms == 35.0


def test_verify_record_hindi_success():
    """Whole-document verification succeeds for a Hindi record with Devanagari text."""
    mock_v = _make_mock_verifier(label="entailment", entail_prob=0.89)
    doc_verifier = WholeDocumentVerifier(verifier=mock_v)

    rec = _make_sample_generation_record(
        record_id="hi_test_01",
        language="hi",
        source_text="भारतीय अंतरिक्ष अनुसंधान संगठन (इसरो) ने चंद्रयान-3 का सफल प्रक्षेपण किया।",
        generated_summary="इसरो ने चंद्रयान-3 सफलतापूर्वक लॉन्च किया।",
    )

    result = doc_verifier.verify_record(rec)

    assert result.status == "SUCCESS"
    assert result.record_id == "hi_test_01"
    assert result.language == "hi"
    assert result.predicted_label == "entailment"
    assert result.entailment_prob == 0.89
    assert result.verification_strategy == "whole_document"


# ---------------------------------------------------------------------------
# Test 3: Entailment, neutral, and contradiction outcomes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "label,entail,neut,contra",
    [
        ("entailment", 0.95, 0.04, 0.01),
        ("neutral", 0.10, 0.85, 0.05),
        ("contradiction", 0.02, 0.03, 0.95),
    ],
)
def test_verify_record_label_outcomes(label, entail, neut, contra):
    """Verifier accurately reflects entailment, neutral, and contradiction predictions."""
    mock_v = _make_mock_verifier(label=label, entail_prob=entail, neut_prob=neut, contra_prob=contra)
    doc_verifier = WholeDocumentVerifier(verifier=mock_v)

    rec = _make_sample_generation_record(record_id=f"test_{label}")
    result = doc_verifier.verify_record(rec)

    assert result.status == "SUCCESS"
    assert result.predicted_label == label
    assert result.entailment_prob == entail
    assert result.neutral_prob == neut
    assert result.contradiction_prob == contra


# ---------------------------------------------------------------------------
# Test 4: Multiple-record batching without reloads
# ---------------------------------------------------------------------------

def test_verify_records_multiple():
    """Multiple records are processed in batch without repeated model loading."""
    mock_v = _make_mock_verifier()
    doc_verifier = WholeDocumentVerifier(verifier=mock_v)

    records = [
        _make_sample_generation_record(record_id=f"rec_{i}", language="en")
        for i in range(5)
    ]

    results = doc_verifier.verify_records(records, batch_size=2)

    assert len(results) == 5
    for i, res in enumerate(results):
        assert res.record_id == f"rec_{i}"
        assert res.status == "SUCCESS"
        assert res.verification_strategy == "whole_document"

    # verify_batch should be invoked across batches rather than 5 individual unbatched loads
    assert mock_v.verify_batch.called


# ---------------------------------------------------------------------------
# Test 5: Empty and invalid inputs
# ---------------------------------------------------------------------------

def test_empty_source_text():
    """Record with empty source text produces FAILED status without crashing."""
    mock_v = _make_mock_verifier()
    doc_verifier = WholeDocumentVerifier(verifier=mock_v)

    rec = _make_sample_generation_record(source_text="")
    result = doc_verifier.verify_record(rec)

    assert result.status == "FAILED"
    assert "Source text" in result.error_message
    assert result.predicted_label == "error"
    assert not mock_v.verify_batch.called


def test_empty_generated_summary():
    """Record with empty generated summary produces FAILED status without crashing."""
    mock_v = _make_mock_verifier()
    doc_verifier = WholeDocumentVerifier(verifier=mock_v)

    rec = _make_sample_generation_record(generated_summary="   ")
    result = doc_verifier.verify_record(rec)

    assert result.status == "FAILED"
    assert "Generated summary" in result.error_message
    assert not mock_v.verify_batch.called


def test_unsupported_language():
    """Record with unsupported language produces FAILED status with descriptive error."""
    mock_v = _make_mock_verifier()
    doc_verifier = WholeDocumentVerifier(verifier=mock_v)

    rec = _make_sample_generation_record(language="klingon")
    result = doc_verifier.verify_record(rec)

    assert result.status == "FAILED"
    assert "Unsupported language" in result.error_message
    assert not mock_v.verify_batch.called


def test_invalid_record_type():
    """Non-GenerationRecord object in sequence is recorded as FAILED."""
    mock_v = _make_mock_verifier()
    doc_verifier = WholeDocumentVerifier(verifier=mock_v)

    results = doc_verifier.verify_records(["invalid_string"])  # type: ignore
    assert len(results) == 1
    assert results[0].status == "FAILED"
    assert "GenerationRecord" in results[0].error_message


# ---------------------------------------------------------------------------
# Test 6: Inference error handling
# ---------------------------------------------------------------------------

def test_inference_error_handling():
    """Inference exceptions from adapter are trapped and recorded as FAILED."""
    mock_v = MagicMock()
    mock_v.config = NLIConfig(model_id="joeddav/xlm-roberta-large-xnli")
    mock_v.verify_batch.side_effect = RuntimeError("CUDA Out of Memory simulation")

    doc_verifier = WholeDocumentVerifier(verifier=mock_v)
    rec = _make_sample_generation_record()

    result = doc_verifier.verify_record(rec)

    assert result.status == "FAILED"
    assert "CUDA Out of Memory" in result.error_message
    assert result.predicted_label == "error"


# ---------------------------------------------------------------------------
# Test 7: Source record preservation
# ---------------------------------------------------------------------------

def test_preservation_of_source_records():
    """Input GenerationRecord instances remain completely unmodified after verification."""
    mock_v = _make_mock_verifier()
    doc_verifier = WholeDocumentVerifier(verifier=mock_v)

    rec = _make_sample_generation_record(record_id="rec_unmodified")
    orig_dict = rec.to_dict()

    _ = doc_verifier.verify_record(rec)

    assert rec.to_dict() == orig_dict
    assert rec.source_text == orig_dict["source_text"]
    assert rec.generated_summary == orig_dict["generated_summary"]


# ---------------------------------------------------------------------------
# Test 8: Serialization & persistence roundtrip
# ---------------------------------------------------------------------------

def test_document_verification_serialization_roundtrip(tmp_path: Path):
    """DocumentVerificationResult to_dict/from_dict and JSONL persistence preserve all data."""
    res1 = DocumentVerificationResult(
        record_id="rec_01",
        language="en",
        predicted_label="entailment",
        entailment_prob=0.91,
        neutral_prob=0.07,
        contradiction_prob=0.02,
        probabilities={"entailment": 0.91, "neutral": 0.07, "contradiction": 0.02},
        model_id="joeddav/xlm-roberta-large-xnli",
        latency_ms=25.4,
        verification_strategy="whole_document",
        status="SUCCESS",
        error_message=None,
        verification_timestamp_utc="2026-11-21T18:30:00Z",
        prompt_strategy="prompt_a_standard",
        source_dataset="csebuetnlp/xlsum",
    )
    res2 = DocumentVerificationResult(
        record_id="rec_02",
        language="hi",
        predicted_label="error",
        entailment_prob=0.0,
        neutral_prob=0.0,
        contradiction_prob=0.0,
        probabilities={},
        model_id="joeddav/xlm-roberta-large-xnli",
        latency_ms=0.0,
        verification_strategy="whole_document",
        status="FAILED",
        error_message="Empty premise",
        verification_timestamp_utc="2026-11-21T18:30:01Z",
        prompt_strategy="prompt_b_factuality",
        source_dataset="csebuetnlp/xlsum",
    )

    out_file = tmp_path / "test_verifications.jsonl"
    count = save_document_verification_results([res1, res2], out_file)
    assert count == 2

    loaded = load_document_verification_results(out_file)
    assert len(loaded) == 2

    assert loaded[0].record_id == res1.record_id
    assert loaded[0].predicted_label == "entailment"
    assert loaded[0].entailment_prob == 0.91
    assert loaded[0].verification_strategy == "whole_document"
    assert loaded[0].status == "SUCCESS"

    assert loaded[1].record_id == res2.record_id
    assert loaded[1].status == "FAILED"
    assert loaded[1].error_message == "Empty premise"
