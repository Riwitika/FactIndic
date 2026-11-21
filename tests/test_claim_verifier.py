"""Unit tests for Hybrid Claim-Level NLI Verification.

Tests run 100% offline using mocked XNLIVerifier outputs without network calls.
Tests:
    - English and Hindi claim verification
    - Verification outcomes: entailment, neutral, and contradiction
    - Multiple evidence passages preserving retrieval rankings
    - No evidence available / zero lexical overlap handling
    - Empty inputs, unsupported languages, and inference failure handling
    - Multiple claims per record and multiple records batching
    - Result ordering, traceability, and character offsets
    - Input preservation
    - JSONL serialization and deserialization roundtrip
"""

from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock
import pytest

from factindic.generation import GenerationRecord
from factindic.verification import (
    BM25EvidenceRetriever,
    ClaimEvidenceResult,
    ClaimVerificationResult,
    ExtractedClaim,
    HybridClaimVerifier,
    NLILabel,
    NLIResult,
    XNLIVerifier,
    load_claim_verification_results,
    save_claim_verification_results,
)


def _make_mock_verifier(
    label: str = "entailment",
    entailment_prob: float = 0.95,
    neutral_prob: float = 0.04,
    contradiction_prob: float = 0.01,
) -> MagicMock:
    """Helper to construct a mocked XNLIVerifier."""
    mock = MagicMock(spec=XNLIVerifier)
    mock.config = MagicMock()
    mock.config.model_id = "joeddav/xlm-roberta-large-xnli"

    def _mock_verify_batch(pairs, language="en", batch_size=None):
        results = []
        for p, h in pairs:
            results.append(
                NLIResult(
                    premise=p,
                    hypothesis=h,
                    language=language,
                    predicted_label=label,
                    entailment_prob=entailment_prob,
                    neutral_prob=neutral_prob,
                    contradiction_prob=contradiction_prob,
                    model_id="joeddav/xlm-roberta-large-xnli",
                    latency_ms=12.5,
                )
            )
        return results

    mock.verify_batch.side_effect = _mock_verify_batch
    return mock


def _make_sample_record(
    record_id: str,
    language: str,
    source_text: str,
) -> GenerationRecord:
    """Helper to construct a valid GenerationRecord."""
    return GenerationRecord(
        source_dataset="csebuetnlp/xlsum",
        source_config=language,
        source_split="test",
        source_record_id=record_id,
        language=language,
        source_text=source_text,
        reference_summary="Reference summary.",
        prompt_strategy="few_shot_prompt_b",
        prompt_template_version="1.0.0",
        model_id="command-a-plus-05-2026",
        provider="cohere",
        generation_parameters={},
        generated_summary="Generated summary.",
        generation_timestamp_utc="2026-11-21T12:00:00Z",
    )


def test_verify_claim_english_entailment():
    source_article = (
        "The Indian Space Research Organisation launched the Chandrayaan-3 lunar mission from Sriharikota on Friday. "
        "The spacecraft entered lunar orbit after a series of successful trajectory adjustments in August. "
        "The Vikram lander made history by touching down safely near the lunar south pole."
    )
    claim = ExtractedClaim(
        record_id="rec_en_01",
        language="en",
        original_sentence_index=0,
        original_sentence_text="ISRO launched Chandrayaan-3 from Sriharikota on Friday.",
        claim_id="rec_en_01_s0_c0",
        atomic_claim_text="ISRO launched Chandrayaan-3 on Friday from Sriharikota.",
    )

    verifier = _make_mock_verifier(
        label="entailment",
        entailment_prob=0.96,
        neutral_prob=0.03,
        contradiction_prob=0.01,
    )
    hybrid = HybridClaimVerifier(verifier=verifier, top_k_evidence=1)
    results = hybrid.verify_claim(claim=claim, source_text=source_article)

    assert len(results) == 1
    res = results[0]
    assert res.record_id == "rec_en_01"
    assert res.claim_id == "rec_en_01_s0_c0"
    assert res.original_sentence_index == 0
    assert res.original_sentence_text == claim.original_sentence_text
    assert res.atomic_claim_text == claim.atomic_claim_text
    assert "launched the Chandrayaan-3" in res.evidence_passage
    assert res.evidence_rank == 1
    assert res.evidence_score > 0.0
    assert source_article[res.evidence_start_offset : res.evidence_end_offset] == res.evidence_passage
    assert res.language == "en"
    assert res.predicted_label == "entailment"
    assert res.entailment_prob == 0.96
    assert res.verification_strategy == "hybrid_claim_level"
    assert res.status == "SUCCESS"


def test_verify_claim_hindi_contradiction():
    source_article = (
        "रिजर्व बैंक ने बुधवार को रेपो दर में 25 आधार अंकों की कटौती की घोषणा की। "
        "वाणिज्यिक बैंकों ने तुरंत ग्राहकों के लिए ऋण की ब्याज दरों में कमी कर दी।"
    )
    claim = ExtractedClaim(
        record_id="rec_hi_01",
        language="hi",
        original_sentence_index=0,
        original_sentence_text="रिजर्व बैंक ने ब्याज दरों में बढ़ोतरी की।",
        claim_id="rec_hi_01_s0_c0",
        atomic_claim_text="रिजर्व बैंक ने रेपो दर में 25 आधार अंकों की बढ़ोतरी की।",
    )

    verifier = _make_mock_verifier(
        label="contradiction",
        entailment_prob=0.02,
        neutral_prob=0.05,
        contradiction_prob=0.93,
    )
    hybrid = HybridClaimVerifier(verifier=verifier, top_k_evidence=1)
    results = hybrid.verify_claim(claim=claim, source_text=source_article)

    assert len(results) == 1
    res = results[0]
    assert res.record_id == "rec_hi_01"
    assert res.language == "hi"
    assert res.predicted_label == "contradiction"
    assert res.contradiction_prob == 0.93
    assert "रेपो दर में 25 आधार अंकों की कटौती" in res.evidence_passage
    assert res.status == "SUCCESS"


def test_verify_claim_multiple_evidence_passages_preserved_ranking():
    source_article = (
        "The telecom commission allocated 5G spectrum bands to three network operators in October. "
        "Commercial services officially rolled out in Delhi, Mumbai, and Bengaluru two months later. "
        "The operators announced attractive data pricing packages for smartphone users."
    )
    claim = ExtractedClaim(
        record_id="rec_multi_ev",
        language="en",
        original_sentence_index=0,
        original_sentence_text="5G spectrum was allocated and services launched in major cities.",
        claim_id="rec_multi_ev_s0_c0",
        atomic_claim_text="5G services rolled out in Delhi, Mumbai, and Bengaluru.",
    )

    verifier = _make_mock_verifier()
    hybrid = HybridClaimVerifier(verifier=verifier, top_k_evidence=2)
    results = hybrid.verify_claim(claim=claim, source_text=source_article)

    assert len(results) == 2
    assert results[0].evidence_rank == 1
    assert results[1].evidence_rank == 2
    assert results[0].evidence_score >= results[1].evidence_score
    # Both passages must trace back to the article
    for r in results:
        assert source_article[r.evidence_start_offset : r.evidence_end_offset] == r.evidence_passage
        assert r.claim_id == "rec_multi_ev_s0_c0"
        assert r.status == "SUCCESS"


def test_verify_claim_no_evidence_available():
    source_article = "Agricultural subsidies support organic farming practices in southern states."
    claim = ExtractedClaim(
        record_id="rec_no_ev",
        language="en",
        original_sentence_index=0,
        original_sentence_text="Astronauts conducted spacewalks outside the space station.",
        claim_id="rec_no_ev_s0_c0",
        atomic_claim_text="Astronauts conducted spacewalks.",
    )

    verifier = _make_mock_verifier()
    hybrid = HybridClaimVerifier(verifier=verifier, top_k_evidence=2)
    results = hybrid.verify_claim(claim=claim, source_text=source_article)

    assert len(results) == 1
    res = results[0]
    assert res.status == "NO_EVIDENCE"
    assert res.predicted_label == "neutral"
    assert res.neutral_prob == 1.0
    assert res.entailment_prob == 0.0
    assert res.evidence_score == 0.0
    # Verifier must NOT have been called for inference
    verifier.verify_batch.assert_not_called()


def test_verify_claim_empty_source_article():
    claim = ExtractedClaim(
        record_id="rec_empty_src",
        language="en",
        original_sentence_index=0,
        original_sentence_text="Some valid claim text.",
        claim_id="rec_empty_src_s0_c0",
        atomic_claim_text="Some valid claim text.",
    )

    verifier = _make_mock_verifier()
    hybrid = HybridClaimVerifier(verifier=verifier)
    results = hybrid.verify_claim(claim=claim, source_text="")

    assert len(results) == 1
    assert results[0].status == "FAILED"
    assert "Source text is empty" in (results[0].error_message or "")


def test_verify_claim_empty_claim_text():
    article = "Valid source article content."
    claim = ExtractedClaim(
        record_id="rec_empty_claim",
        language="en",
        original_sentence_index=0,
        original_sentence_text="Some sentence text.",
        claim_id="rec_empty_claim_s0_c0",
        atomic_claim_text="",
    )

    verifier = _make_mock_verifier()
    hybrid = HybridClaimVerifier(verifier=verifier)
    results = hybrid.verify_claim(claim=claim, source_text=article)

    assert len(results) == 1
    assert results[0].status == "FAILED"
    assert "Atomic claim text is empty" in (results[0].error_message or "")


def test_verify_claim_unsupported_language():
    article = "Some French text here."
    claim = ExtractedClaim(
        record_id="rec_fr_01",
        language="fr",
        original_sentence_index=0,
        original_sentence_text="Un fait ici.",
        claim_id="rec_fr_01_s0_c0",
        atomic_claim_text="Un fait ici.",
    )

    verifier = _make_mock_verifier()
    hybrid = HybridClaimVerifier(verifier=verifier)
    results = hybrid.verify_claim(claim=claim, source_text=article)

    assert len(results) == 1
    assert results[0].status == "FAILED"
    assert "Unsupported language" in (results[0].error_message or "")


def test_verify_claim_inference_failure_handling():
    source_article = "The government announced substantial education grant increments."
    claim = ExtractedClaim(
        record_id="rec_err_01",
        language="en",
        original_sentence_index=0,
        original_sentence_text="Education grants were increased.",
        claim_id="rec_err_01_s0_c0",
        atomic_claim_text="Education grants were increased by the government.",
    )

    verifier = MagicMock(spec=XNLIVerifier)
    verifier.config = MagicMock()
    verifier.config.model_id = "joeddav/xlm-roberta-large-xnli"
    verifier.verify_batch.side_effect = RuntimeError("CUDA out of memory exception")

    hybrid = HybridClaimVerifier(verifier=verifier, top_k_evidence=1)
    results = hybrid.verify_claim(claim=claim, source_text=source_article)

    assert len(results) == 1
    assert results[0].status == "FAILED"
    assert results[0].predicted_label == "error"
    assert "RuntimeError" in (results[0].error_message or "")


def test_verify_record_multiple_claims_and_preservation():
    record = _make_sample_record(
        record_id="rec_rec_batch",
        language="en",
        source_text=(
            "The Department of Renewable Energy sanctioned 10 gigawatts of solar park projects. "
            "Completion is targeted for December 2027 across four western states. "
            "Initial environmental impact assessments received clearance."
        ),
    )

    claims = [
        ExtractedClaim(
            record_id="rec_rec_batch",
            language="en",
            original_sentence_index=0,
            original_sentence_text="The ministry sanctioned 10 gigawatts of solar projects.",
            claim_id="rec_rec_batch_s0_c0",
            atomic_claim_text="The Department sanctioned 10 gigawatts of solar park projects.",
        ),
        ExtractedClaim(
            record_id="rec_rec_batch",
            language="en",
            original_sentence_index=1,
            original_sentence_text="Projects will finish in December 2027.",
            claim_id="rec_rec_batch_s1_c0",
            atomic_claim_text="Completion of projects is targeted for December 2027.",
        ),
    ]

    verifier = _make_mock_verifier(label="entailment", entailment_prob=0.92)
    hybrid = HybridClaimVerifier(verifier=verifier, top_k_evidence=1)
    results = hybrid.verify_record(record=record, claims=claims)

    assert len(results) == 2
    assert results[0].claim_id == "rec_rec_batch_s0_c0"
    assert results[1].claim_id == "rec_rec_batch_s1_c0"
    assert results[0].predicted_label == "entailment"
    assert results[1].predicted_label == "entailment"

    # Verify input objects were NOT modified
    assert record.source_record_id == "rec_rec_batch"
    assert claims[0].claim_id == "rec_rec_batch_s0_c0"
    assert claims[1].claim_id == "rec_rec_batch_s1_c0"


def test_verify_records_batch_multilingual():
    record_en = _make_sample_record(
        record_id="rec_batch_en",
        language="en",
        source_text="Clean energy projects expanded rapidly across the country in 2024.",
    )
    record_hi = _make_sample_record(
        record_id="rec_batch_hi",
        language="hi",
        source_text="देश भर में नई सिंचाई योजनाओं को मंजूरी दी गई है।",
    )

    claims = [
        ExtractedClaim(
            record_id="rec_batch_en",
            language="en",
            original_sentence_index=0,
            original_sentence_text="Clean energy projects expanded in 2024.",
            claim_id="rec_batch_en_s0_c0",
            atomic_claim_text="Clean energy projects expanded rapidly in 2024.",
        ),
        ExtractedClaim(
            record_id="rec_batch_hi",
            language="hi",
            original_sentence_index=0,
            original_sentence_text="सिंचाई योजनाओं को मंजूरी मिली।",
            claim_id="rec_batch_hi_s0_c0",
            atomic_claim_text="देश भर में नई सिंचाई योजनाओं को मंजूरी दी गई है।",
        ),
    ]

    verifier = _make_mock_verifier(label="entailment", entailment_prob=0.91)
    hybrid = HybridClaimVerifier(verifier=verifier, top_k_evidence=1)
    results = hybrid.verify_records(records=[record_en, record_hi], claims=claims)

    assert len(results) == 2
    assert results[0].record_id == "rec_batch_en"
    assert results[0].language == "en"
    assert results[1].record_id == "rec_batch_hi"
    assert results[1].language == "hi"


def test_claim_verification_serialization_roundtrip(tmp_path: Path):
    result = ClaimVerificationResult(
        record_id="rec_ser_01",
        claim_id="rec_ser_01_s0_c0",
        original_sentence_index=0,
        original_sentence_text="ISRO launched Chandrayaan-3.",
        atomic_claim_text="ISRO launched Chandrayaan-3 from Sriharikota.",
        evidence_passage="ISRO launched Chandrayaan-3 from Sriharikota on Friday.",
        evidence_start_offset=0,
        evidence_end_offset=56,
        evidence_rank=1,
        evidence_score=3.456,
        language="en",
        predicted_label="entailment",
        entailment_prob=0.95,
        neutral_prob=0.04,
        contradiction_prob=0.01,
        probabilities={"contradiction": 0.01, "neutral": 0.04, "entailment": 0.95},
        model_id="joeddav/xlm-roberta-large-xnli",
        latency_ms=15.2,
        verification_strategy="hybrid_claim_level",
        status="SUCCESS",
        error_message=None,
        verification_timestamp_utc="2026-11-21T12:00:00Z",
        prompt_strategy="few_shot_prompt_b",
        source_dataset="csebuetnlp/xlsum",
    )

    file_path = tmp_path / "claim_verification_results.jsonl"
    save_claim_verification_results([result], file_path)

    loaded = load_claim_verification_results(file_path)
    assert len(loaded) == 1
    loaded_result = loaded[0]

    assert loaded_result.record_id == result.record_id
    assert loaded_result.claim_id == result.claim_id
    assert loaded_result.original_sentence_index == result.original_sentence_index
    assert loaded_result.original_sentence_text == result.original_sentence_text
    assert loaded_result.atomic_claim_text == result.atomic_claim_text
    assert loaded_result.evidence_passage == result.evidence_passage
    assert loaded_result.evidence_start_offset == result.evidence_start_offset
    assert loaded_result.evidence_end_offset == result.evidence_end_offset
    assert loaded_result.evidence_rank == result.evidence_rank
    assert loaded_result.evidence_score == result.evidence_score
    assert loaded_result.language == result.language
    assert loaded_result.predicted_label == result.predicted_label
    assert loaded_result.entailment_prob == result.entailment_prob
    assert loaded_result.neutral_prob == result.neutral_prob
    assert loaded_result.contradiction_prob == result.contradiction_prob
    assert loaded_result.model_id == result.model_id
    assert loaded_result.verification_strategy == result.verification_strategy
    assert loaded_result.status == result.status
