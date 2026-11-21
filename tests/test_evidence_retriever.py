"""Unit tests for Claim-Level Evidence Retrieval (BM25 lexical passage ranking).

Tests:
    - Multilingual tokenization (English and Hindi Devanagari).
    - Source article passage segmentation with exact character offsets.
    - BM25Okapi scoring and rank ordering for English and Hindi.
    - ClaimEvidenceResult schema, attributes, and serialization roundtrip.
    - Handling of empty source articles, missing claims, and zero-score/no-evidence cases.
    - Non-destructive preservation of GenerationRecord and ExtractedClaim attributes.
    - JSONL persistence roundtrip.
"""

import json
from pathlib import Path
import pytest

from factindic.generation import GenerationRecord
from factindic.verification import (
    BM25EvidenceRetriever,
    ClaimEvidenceResult,
    EvidencePassage,
    ExtractedClaim,
    load_evidence_results,
    save_evidence_results,
    segment_article_into_passages,
    tokenize_text,
)


def test_tokenize_text_english():
    text = "ISRO successfully launched the Chandrayaan-3 mission in July 2023."
    tokens = tokenize_text(text, language="en")
    assert "isro" in tokens
    assert "chandrayaan" in tokens
    assert "3" in tokens
    assert "2023" in tokens


def test_tokenize_text_hindi():
    text = "इसरो ने जुलाई 2023 में चंद्रयान-3 मिशन सफलतापूर्वक लॉन्च किया।"
    tokens = tokenize_text(text, language="hi")
    assert "इसर" in tokens or "इसरो" in tokens
    assert "च द रय न" in " ".join(tokens) or "चंद्रयान" in tokens
    assert "लॉन्च" in tokens
    assert "2023" in tokens


def test_segment_article_passages_english_offsets():
    article = (
        "ISRO launched Chandrayaan-3 from Sriharikota. "
        "The spacecraft entered lunar orbit in August. "
        "The lander touched down safely near the south pole."
    )
    passages = segment_article_into_passages(article, language="en")
    assert len(passages) == 3

    for p in passages:
        # Crucial check: character slice into source text must exactly equal passage text
        extracted_slice = article[p.start_char : p.end_char]
        assert extracted_slice == p.text, f"Mismatch: '{extracted_slice}' vs '{p.text}'"

    assert passages[0].text == "ISRO launched Chandrayaan-3 from Sriharikota."
    assert passages[1].text == "The spacecraft entered lunar orbit in August."
    assert passages[2].text == "The lander touched down safely near the south pole."


def test_segment_article_passages_hindi_offsets():
    article = (
        "इसरो ने श्रीहरिकोटा से चंद्रयान-3 लॉन्च किया। "
        "अंतरिक्ष यान अगस्त में चंद्रमा की कक्षा में पहुंचा। "
        "लैंडर सुरक्षित रूप से दक्षिणी ध्रुव पर उतरा।"
    )
    passages = segment_article_into_passages(article, language="hi")
    assert len(passages) == 3

    for p in passages:
        extracted_slice = article[p.start_char : p.end_char]
        assert extracted_slice == p.text, f"Mismatch: '{extracted_slice}' vs '{p.text}'"

    assert passages[0].text == "इसरो ने श्रीहरिकोटा से चंद्रयान-3 लॉन्च किया।"
    assert passages[1].text == "अंतरिक्ष यान अगस्त में चंद्रमा की कक्षा में पहुंचा।"
    assert passages[2].text == "लैंडर सुरक्षित रूप से दक्षिणी ध्रुव पर उतरा।"


def test_retrieve_for_claim_english_ranking():
    article = (
        "The Reserve Bank of India raised repo rates by 25 basis points on Wednesday. "
        "Commercial banks immediately increased home loan interest rates across all tiers. "
        "The automotive sector reported a slight decline in quarterly passenger car sales."
    )
    claim = ExtractedClaim(
        record_id="rec_en_01",
        language="en",
        original_sentence_index=0,
        original_sentence_text="The central bank raised repo rates by 25 basis points.",
        claim_id="rec_en_01_s0_c0",
        atomic_claim_text="The Reserve Bank raised repo rates by 25 basis points.",
    )

    retriever = BM25EvidenceRetriever()
    results = retriever.retrieve_for_claim(claim, source_text=article, top_k=2)

    assert len(results) >= 1
    top_result = results[0]
    assert top_result.retrieval_status == "SUCCESS"
    assert top_result.retrieval_rank == 1
    assert top_result.record_id == "rec_en_01"
    assert top_result.claim_id == "rec_en_01_s0_c0"
    assert top_result.language == "en"
    assert "Reserve Bank of India raised repo rates" in top_result.evidence_passage
    assert article[top_result.evidence_start_offset : top_result.evidence_end_offset] == top_result.evidence_passage
    assert top_result.relevance_score > 0.0


def test_retrieve_for_claim_hindi_ranking():
    article = (
        "भारतीय अंतरिक्ष अनुसंधान संगठन (इसरो) ने शुक्रवार को एक नया मौसम उपग्रह लॉन्च किया। "
        "इस उपग्रह का मुख्य उद्देश्य चक्रवातों और मानसून की सटीक भविष्यवाणी करना है। "
        "मौसम विज्ञान विभाग ने इस सफलता का स्वागत करते हुए इसे मील का पत्थर बताया।"
    )
    claim = ExtractedClaim(
        record_id="rec_hi_01",
        language="hi",
        original_sentence_index=0,
        original_sentence_text="उपग्रह चक्रवातों और मानसून की भविष्यवाणी करेगा।",
        claim_id="rec_hi_01_s0_c0",
        atomic_claim_text="उपग्रह का उद्देश्य चक्रवातों और मानसून की सटीक भविष्यवाणी करना है।",
    )

    retriever = BM25EvidenceRetriever()
    results = retriever.retrieve_for_claim(claim, source_text=article, top_k=2)

    assert len(results) >= 1
    top_result = results[0]
    assert top_result.retrieval_status == "SUCCESS"
    assert top_result.retrieval_rank == 1
    assert "चक्रवातों और मानसून की सटीक भविष्यवाणी" in top_result.evidence_passage
    assert article[top_result.evidence_start_offset : top_result.evidence_end_offset] == top_result.evidence_passage
    assert top_result.relevance_score > 0.0


def _make_sample_record(
    record_id: str,
    language: str,
    source_text: str,
) -> GenerationRecord:
    return GenerationRecord(
        source_dataset="csebuetnlp/xlsum",
        source_config=language,
        source_split="test",
        source_record_id=record_id,
        language=language,
        source_text=source_text,
        reference_summary="Reference summary.",
        prompt_strategy="prompt_a",
        prompt_template_version="1.0.0",
        model_id="command-a-plus-05-2026",
        provider="cohere",
        generation_parameters={},
        generated_summary="Generated summary.",
        generation_timestamp_utc="2026-11-21T12:00:00Z",
    )


def test_retrieve_no_evidence_zero_overlap():
    # Ensure zero token overlap between article and claim
    article = "Agricultural subsidies support organic farming practices."
    claim = ExtractedClaim(
        record_id="rec_unrelated_01",
        language="en",
        original_sentence_index=0,
        original_sentence_text="Astronauts conducted spacewalks.",
        claim_id="rec_unrelated_01_s0_c0",
        atomic_claim_text="Astronauts conducted spacewalks.",
    )

    retriever = BM25EvidenceRetriever()
    results = retriever.retrieve_for_claim(claim, source_text=article, top_k=1)

    assert len(results) == 1
    assert results[0].retrieval_status == "NO_EVIDENCE"
    assert results[0].relevance_score == 0.0


def test_retrieve_empty_source_article():
    claim = ExtractedClaim(
        record_id="rec_empty_src",
        language="en",
        original_sentence_index=0,
        original_sentence_text="Some claim text.",
        claim_id="rec_empty_src_s0_c0",
        atomic_claim_text="Some claim text.",
    )

    retriever = BM25EvidenceRetriever()
    results = retriever.retrieve_for_claim(claim, source_text="", top_k=2)

    assert len(results) == 1
    assert results[0].retrieval_status == "FAILED"
    assert "empty or missing" in (results[0].error_message or "")


def test_retrieve_empty_claim_text():
    article = "Valid source text about climate change."
    claim = ExtractedClaim(
        record_id="rec_empty_claim",
        language="en",
        original_sentence_index=0,
        original_sentence_text="",
        claim_id="rec_empty_claim_s0_c0",
        atomic_claim_text="",
    )

    retriever = BM25EvidenceRetriever()
    results = retriever.retrieve_for_claim(claim, source_text=article, top_k=2)

    assert len(results) == 1
    assert results[0].retrieval_status == "FAILED"
    assert "claim text is empty" in (results[0].error_message or "")


def test_retrieve_unsupported_language():
    claim = ExtractedClaim(
        record_id="rec_lang_err",
        language="fr",
        original_sentence_index=0,
        original_sentence_text="Le président a parlé.",
        claim_id="rec_lang_err_s0_c0",
        atomic_claim_text="Le président a parlé.",
    )

    retriever = BM25EvidenceRetriever()
    results = retriever.retrieve_for_claim(claim, source_text="Quelque texte ici.", top_k=2)

    assert len(results) == 1
    assert results[0].retrieval_status == "FAILED"
    assert "Unsupported language" in (results[0].error_message or "")


def test_retrieve_for_record_preservation_and_batching():
    record = _make_sample_record(
        record_id="rec_multi_01",
        language="en",
        source_text=(
            "The Department of Telecommunications allocated 5G spectrum bands to three operators. "
            "Commercial rollout began in major metropolitan areas in October. "
            "Tariff plans were announced with introductory discounts for early adopters."
        ),
    )

    claims = [
        ExtractedClaim(
            record_id="rec_multi_01",
            language="en",
            original_sentence_index=0,
            original_sentence_text="5G spectrum was allocated.",
            claim_id="rec_multi_01_s0_c0",
            atomic_claim_text="5G spectrum was allocated to three operators.",
        ),
        ExtractedClaim(
            record_id="rec_multi_01",
            language="en",
            original_sentence_index=1,
            original_sentence_text="Rollouts began in October.",
            claim_id="rec_multi_01_s1_c0",
            atomic_claim_text="Commercial rollout began in October in major metropolitan areas.",
        ),
    ]

    retriever = BM25EvidenceRetriever()
    results = retriever.retrieve_for_record(record, claims=claims, top_k=2)

    # 2 claims * up to 2 passages = up to 4 results
    assert len(results) >= 2
    claim_ids = {r.claim_id for r in results}
    assert "rec_multi_01_s0_c0" in claim_ids
    assert "rec_multi_01_s1_c0" in claim_ids

    # Verify original record and claim remain unmodified
    assert record.source_record_id == "rec_multi_01"
    assert len(record.source_text) > 0
    assert claims[0].claim_id == "rec_multi_01_s0_c0"


def test_retrieve_for_records_multiple():
    record1 = _make_sample_record(
        record_id="rec_batch_01",
        language="en",
        source_text="First source article about clean energy initiatives in solar power.",
    )
    record2 = _make_sample_record(
        record_id="rec_batch_02",
        language="hi",
        source_text="दूसरा लेख कृषि और सिंचाई परियोजनाओं के बारे में है।",
    )

    claims = [
        ExtractedClaim(
            record_id="rec_batch_01",
            language="en",
            original_sentence_index=0,
            original_sentence_text="Solar power initiatives were introduced.",
            claim_id="rec_batch_01_s0_c0",
            atomic_claim_text="Solar power initiatives were introduced.",
        ),
        ExtractedClaim(
            record_id="rec_batch_02",
            language="hi",
            original_sentence_index=0,
            original_sentence_text="सिंचाई परियोजनाओं की शुरुआत हुई।",
            claim_id="rec_batch_02_s0_c0",
            atomic_claim_text="सिंचाई परियोजनाओं की शुरुआत हुई।",
        ),
    ]

    retriever = BM25EvidenceRetriever()
    results = retriever.retrieve_for_records([record1, record2], claims=claims, top_k=1)

    assert len(results) == 2
    assert results[0].record_id == "rec_batch_01"
    assert results[0].language == "en"
    assert results[1].record_id == "rec_batch_02"
    assert results[1].language == "hi"


def test_evidence_result_serialization_roundtrip(tmp_path: Path):
    result = ClaimEvidenceResult(
        record_id="rec_ser_01",
        claim_id="rec_ser_01_s0_c0",
        original_claim_text="चंद्रयान-3 मिशन चंद्रमा पर पहुंचा।",
        evidence_passage="अंतरिक्ष यान अगस्त में चंद्रमा की कक्षा में पहुंचा।",
        evidence_start_offset=46,
        evidence_end_offset=98,
        retrieval_rank=1,
        relevance_score=2.3456,
        language="hi",
        retrieval_strategy="bm25_lexical",
        retrieval_status="SUCCESS",
        error_message=None,
        retrieval_timestamp_utc="2026-11-21T12:00:00Z",
    )

    file_path = tmp_path / "evidence_results.jsonl"
    save_evidence_results([result], file_path)

    loaded = load_evidence_results(file_path)
    assert len(loaded) == 1
    loaded_result = loaded[0]

    assert loaded_result.record_id == result.record_id
    assert loaded_result.claim_id == result.claim_id
    assert loaded_result.original_claim_text == result.original_claim_text
    assert loaded_result.evidence_passage == result.evidence_passage
    assert loaded_result.evidence_start_offset == result.evidence_start_offset
    assert loaded_result.evidence_end_offset == result.evidence_end_offset
    assert loaded_result.retrieval_rank == result.retrieval_rank
    assert loaded_result.relevance_score == result.relevance_score
    assert loaded_result.language == result.language
    assert loaded_result.retrieval_strategy == result.retrieval_strategy
    assert loaded_result.retrieval_status == result.retrieval_status
