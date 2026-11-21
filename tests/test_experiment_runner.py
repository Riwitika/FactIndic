"""Unit tests for the Unified Six-Condition Experiment Pipeline (C1-C6).

Tests run 100% offline with mocked LLM generation and NLI verification.
Tests:
    - Execution of all six conditions (C1 to C6)
    - Consistency across English and Hindi source records
    - Generation parameter parity across prompt strategies
    - Accurate routing to Whole-Document, Sentence-Level, and Hybrid Claim-Level verifiers
    - Resumption without duplicate runs
    - Failure handling without silent success
    - Output serialization and deserialization
    - Preservation of previous experiment artifacts
"""

import json
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock
import pytest

from factindic.experiment_runner import (
    ConditionID,
    ExperimentRecord,
    UnifiedExperimentRunner,
    load_experiment_records,
    save_experiment_records,
)
from factindic.generation import GenerationRecord
from factindic.models.base import BaseLLMClient, GenerationResult
from factindic.prompts.templates import PROMPT_TEMPLATE_VERSION, PromptStrategy
from factindic.schema import FactIndicRecord
from factindic.verification import (
    AtomicClaimExtractor,
    BM25EvidenceRetriever,
    ClaimVerificationResult,
    DocumentVerificationResult,
    ExtractedClaim,
    HybridClaimVerifier,
    NLILabel,
    NLIResult,
    SentenceLevelVerifier,
    SentenceVerificationResult,
    WholeDocumentVerifier,
    XNLIVerifier,
)


class MockLLMClient(BaseLLMClient):
    """Deterministic mock client returning valid summaries and claim JSONs."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.call_count = 0

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_new_tokens: int = 256,
        top_p: float = 0.9,
    ) -> GenerationResult:
        self.call_count += 1
        if self.should_fail:
            raise RuntimeError("Mock LLM generation failure.")

        # If this is a claim extraction prompt, return JSON array of claims
        if "atomic factual claim" in prompt or "तथ्यात्मक दावों" in prompt:
            return GenerationResult(
                text='["First atomic claim.", "Second atomic claim."]',
                model_id="command-a-plus-05-2026",
                provider="cohere",
                generation_parameters={"temperature": temperature},
                latency_seconds=0.1,
            )

        # Standard summarization response
        return GenerationResult(
            text="First sentence of summary. Second sentence of summary.",
            model_id="command-a-plus-05-2026",
            provider="cohere",
            generation_parameters={
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "top_p": top_p,
            },
            latency_seconds=0.2,
        )


def _make_mock_nli_verifier() -> MagicMock:
    """Mock XNLIVerifier returning standard entailment predictions."""
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
                    predicted_label="entailment",
                    entailment_prob=0.92,
                    neutral_prob=0.06,
                    contradiction_prob=0.02,
                    model_id="joeddav/xlm-roberta-large-xnli",
                    latency_ms=10.0,
                )
            )
        return results

    mock.verify_batch.side_effect = _mock_verify_batch
    return mock


def _make_dummy_factindic_record(
    raw_id: str,
    language: str = "en",
    source_text: str = "First sentence of article. Second sentence of article. Third sentence.",
) -> FactIndicRecord:
    return FactIndicRecord(
        example_id=f"xlsum_{language}_{raw_id}",
        language=language,
        source_text=source_text,
        reference_summary="Ground truth reference summary.",
        metadata={
            "raw_id": raw_id,
            "source_dataset": "csebuetnlp/xlsum",
            "split": "test",
        },
    )


def test_run_all_six_conditions_english():
    record = _make_dummy_factindic_record("doc_en_101", language="en")
    llm = MockLLMClient()
    nli = _make_mock_nli_verifier()

    runner = UnifiedExperimentRunner(
        llm_client=llm,
        nli_verifier=nli,
        delay_seconds=0.0,
    )

    results = runner.run_all_conditions([record], resume=False)

    assert len(results) == 6
    condition_ids = [r.condition_id for r in results]
    assert condition_ids == ["C1", "C2", "C3", "C4", "C5", "C6"]

    # Parameter parity and verification routing check
    c1 = next(r for r in results if r.condition_id == "C1")
    assert c1.prompt_strategy == "prompt_a"
    assert c1.verification_strategy == "whole_document"
    assert c1.status == "SUCCESS"
    assert len(c1.verification_results) == 1

    c2 = next(r for r in results if r.condition_id == "C2")
    assert c2.prompt_strategy == "prompt_b"
    assert c2.verification_strategy == "whole_document"

    c3 = next(r for r in results if r.condition_id == "C3")
    assert c3.prompt_strategy == "prompt_a"
    assert c3.verification_strategy == "sentence_level"
    assert len(c3.verification_results) == 2  # 2 sentences

    c4 = next(r for r in results if r.condition_id == "C4")
    assert c4.prompt_strategy == "prompt_b"
    assert c4.verification_strategy == "sentence_level"

    c5 = next(r for r in results if r.condition_id == "C5")
    assert c5.prompt_strategy == "prompt_a"
    assert c5.verification_strategy == "hybrid_claim_level"
    assert len(c5.verification_results) >= 1

    c6 = next(r for r in results if r.condition_id == "C6")
    assert c6.prompt_strategy == "prompt_b"
    assert c6.verification_strategy == "hybrid_claim_level"


def test_run_conditions_hindi():
    record = _make_dummy_factindic_record(
        "doc_hi_101",
        language="hi",
        source_text="पहला वाक्य। दूसरा वाक्य। तीसरा वाक्य।",
    )
    llm = MockLLMClient()
    nli = _make_mock_nli_verifier()

    runner = UnifiedExperimentRunner(
        llm_client=llm,
        nli_verifier=nli,
        delay_seconds=0.0,
    )

    results = runner.run_all_conditions([record], resume=False)
    assert len(results) == 6
    for r in results:
        assert r.language == "hi"
        assert r.source_record_id == "doc_hi_101"
        assert r.status == "SUCCESS"


def test_generation_reuse_and_parameter_parity():
    record = _make_dummy_factindic_record("doc_parity_01", language="en")
    llm = MockLLMClient()
    nli = _make_mock_nli_verifier()

    runner = UnifiedExperimentRunner(
        llm_client=llm,
        nli_verifier=nli,
        delay_seconds=0.0,
    )

    results = runner.run_all_conditions([record], resume=False)

    # C1, C3, and C5 must share identical Prompt A generation
    c1 = next(r for r in results if r.condition_id == "C1")
    c3 = next(r for r in results if r.condition_id == "C3")
    c5 = next(r for r in results if r.condition_id == "C5")
    assert c1.generated_summary == c3.generated_summary == c5.generated_summary
    assert c1.generation_parameters == c3.generation_parameters == c5.generation_parameters

    # C2, C4, and C6 must share identical Prompt B generation
    c2 = next(r for r in results if r.condition_id == "C2")
    c4 = next(r for r in results if r.condition_id == "C4")
    c6 = next(r for r in results if r.condition_id == "C6")
    assert c2.generated_summary == c4.generated_summary == c6.generated_summary


def test_resumption_and_duplicate_prevention(tmp_path: Path):
    record = _make_dummy_factindic_record("doc_resume_01", language="en")
    llm = MockLLMClient()
    nli = _make_mock_nli_verifier()
    out_file = tmp_path / "exp_results.jsonl"

    runner = UnifiedExperimentRunner(
        llm_client=llm,
        nli_verifier=nli,
        delay_seconds=0.0,
    )

    # First run: executes all 6 conditions
    res1 = runner.run_all_conditions([record], output_path=out_file, resume=True)
    assert len(res1) == 6
    loaded1 = load_experiment_records(out_file)
    assert len(loaded1) == 6

    # Track call count before second run
    llm_calls_after_run1 = llm.call_count

    # Second run with resume=True: should skip all 6 conditions
    res2 = runner.run_all_conditions([record], output_path=out_file, resume=True)
    assert len(res2) == 6
    # No new generation calls made
    assert llm.call_count == llm_calls_after_run1


def test_generation_failure_handling():
    record = _make_dummy_factindic_record("doc_fail_01", language="en")
    llm = MockLLMClient(should_fail=True)
    nli = _make_mock_nli_verifier()

    runner = UnifiedExperimentRunner(
        llm_client=llm,
        nli_verifier=nli,
        delay_seconds=0.0,
    )

    results = runner.run_all_conditions([record], resume=False)
    assert len(results) == 6
    for r in results:
        assert r.status == "FAILED"
        assert "Generation failed" in (r.error_message or "")
        assert len(r.verification_results) == 0


def test_experiment_serialization_roundtrip(tmp_path: Path):
    record = ExperimentRecord(
        condition_id="C5",
        source_record_id="rec_ser_101",
        language="en",
        prompt_strategy="prompt_a",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        verification_strategy="hybrid_claim_level",
        model_id="command-a-plus-05-2026",
        provider="cohere",
        generation_parameters={"temperature": 0.3},
        generated_summary="Summary text.",
        generation_latency_seconds=0.45,
        verification_latency_ms=25.0,
        verification_results=[{"claim_id": "c1", "predicted_label": "entailment"}],
        status="SUCCESS",
        error_message=None,
        timestamp_utc="2026-11-21T12:00:00Z",
        source_dataset="csebuetnlp/xlsum",
    )

    file_path = tmp_path / "experiment_records.jsonl"
    save_experiment_records([record], file_path)

    loaded = load_experiment_records(file_path)
    assert len(loaded) == 1
    loaded_rec = loaded[0]

    assert loaded_rec.condition_id == "C5"
    assert loaded_rec.source_record_id == "rec_ser_101"
    assert loaded_rec.prompt_strategy == "prompt_a"
    assert loaded_rec.verification_strategy == "hybrid_claim_level"
    assert loaded_rec.status == "SUCCESS"
    assert len(loaded_rec.verification_results) == 1


def test_previous_pilot_artifacts_preservation():
    pilot_file = Path("data/generations/controlled_generation_results.jsonl")
    if pilot_file.exists():
        with open(pilot_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 20, "Existing pilot 20-summary dataset must remain intact."
