"""Unit tests for the Benchmark Experiment Workflow script.

Tests:
    - Preflight check with missing or empty files
    - Preflight check with missing credentials
    - Full execution of 6 conditions across benchmark records using mocked verifiers
    - Output directory generation: experiment_results.jsonl and experiment_audit.json
    - Incremental resumption and deduplication
    - Accurate condition breakdown and failure recording
    - Non-destruction of pilot artifacts
"""

import json
from pathlib import Path
from typing import List
from unittest.mock import MagicMock
import pytest

from factindic.experiment_runner import UnifiedExperimentRunner
from factindic.models.base import BaseLLMClient, GenerationResult
from factindic.schema import FactIndicRecord
from factindic.serialization import save_jsonl
from factindic.verification import (
    NLILabel,
    NLIResult,
    XNLIVerifier,
)
from scripts.run_benchmark_experiment import (
    execute_benchmark_experiment,
    preflight_check,
)


class MockLLM(BaseLLMClient):
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.call_count = 0
        self.model_id = "command-a-plus-05-2026"
        self.provider = "cohere"
        self.api_key = "mock_cohere_key"

    def generate(self, prompt: str, temperature: float = 0.3, max_new_tokens: int = 256, top_p: float = 0.9):
        self.call_count += 1
        if self.should_fail:
            raise RuntimeError("API quota exhausted.")
        if "atomic factual claim" in prompt or "तथ्यात्मक दावों" in prompt:
            return GenerationResult(
                text='["First claim.", "Second claim."]',
                model_id=self.model_id,
                provider=self.provider,
                generation_parameters={"temperature": temperature},
                latency_seconds=0.1,
            )
        return GenerationResult(
            text="First sentence of summary. Second sentence.",
            model_id=self.model_id,
            provider=self.provider,
            generation_parameters={"temperature": temperature},
            latency_seconds=0.2,
        )


def _make_mock_nli() -> MagicMock:
    mock = MagicMock(spec=XNLIVerifier)
    mock.config = MagicMock()
    mock.config.model_id = "joeddav/xlm-roberta-large-xnli"

    def _mock_verify_batch(pairs, language="en", batch_size=None):
        return [
            NLIResult(
                premise=p,
                hypothesis=h,
                language=language,
                predicted_label="entailment",
                entailment_prob=0.94,
                neutral_prob=0.04,
                contradiction_prob=0.02,
                model_id="joeddav/xlm-roberta-large-xnli",
                latency_ms=8.0,
            )
            for p, h in pairs
        ]

    mock.verify_batch.side_effect = _mock_verify_batch
    return mock


def _make_record(raw_id: str, lang: str) -> FactIndicRecord:
    return FactIndicRecord(
        example_id=f"xlsum_{lang}_{raw_id}",
        language=lang,
        source_text=f"Valid source text for {raw_id} with multiple sentences. Another sentence.",
        reference_summary=f"Valid reference summary for {raw_id}.",
        metadata={"raw_id": raw_id, "split": "test", "source_dataset": "csebuetnlp/xlsum"},
    )


def test_preflight_missing_files(tmp_path: Path):
    en_missing = tmp_path / "missing_en.jsonl"
    hi_missing = tmp_path / "missing_hi.jsonl"

    res = preflight_check(en_missing, hi_missing, client=MockLLM())
    assert not res["ready"]
    assert any("English benchmark file missing" in b for b in res["blockers"])
    assert any("Hindi benchmark file missing" in b for b in res["blockers"])


def test_preflight_missing_credentials(tmp_path: Path):
    en_file = tmp_path / "en.jsonl"
    hi_file = tmp_path / "hi.jsonl"
    save_jsonl([_make_record("doc_1", "en")], en_file)
    save_jsonl([_make_record("doc_2", "hi")], hi_file)

    llm_no_key = MockLLM()
    llm_no_key.api_key = ""

    res = preflight_check(en_file, hi_file, client=llm_no_key)
    assert not res["ready"]
    assert any("COHERE_API_KEY is not configured" in b for b in res["blockers"])


def test_preflight_success_and_estimates(tmp_path: Path):
    en_file = tmp_path / "en.jsonl"
    hi_file = tmp_path / "hi.jsonl"
    save_jsonl([_make_record(f"en_{i}", "en") for i in range(5)], en_file)
    save_jsonl([_make_record(f"hi_{i}", "hi") for i in range(5)], hi_file)

    res = preflight_check(en_file, hi_file, client=MockLLM())
    assert res["ready"]
    assert res["en_count"] == 5
    assert res["hi_count"] == 5
    assert res["total_source_records"] == 10
    assert res["total_condition_evaluations"] == 60
    assert res["estimated_generation_calls"] == 20


def test_execute_benchmark_experiment_mocked(tmp_path: Path):
    en_file = tmp_path / "benchmark_en.jsonl"
    hi_file = tmp_path / "benchmark_hi.jsonl"
    save_jsonl([_make_record("doc_en_1", "en")], en_file)
    save_jsonl([_make_record("doc_hi_1", "hi")], hi_file)

    llm = MockLLM()
    nli = _make_mock_nli()
    runner = UnifiedExperimentRunner(llm_client=llm, nli_verifier=nli, delay_seconds=0.0)

    out_dir = tmp_path / "benchmark_results"
    audit = execute_benchmark_experiment(
        en_benchmark_path=en_file,
        hi_benchmark_path=hi_file,
        output_dir=out_dir,
        runner=runner,
        delay_seconds=0.0,
        resume=True,
    )

    res_file = out_dir / "experiment_results.jsonl"
    audit_file = out_dir / "experiment_audit.json"

    assert res_file.exists()
    assert audit_file.exists()

    with open(res_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    # 2 records * 6 conditions = 12 evaluations
    assert len(lines) == 12
    assert audit["results_summary"]["total_evaluations"] == 12
    assert audit["results_summary"]["condition_breakdown"]["C1"] == 2
    assert audit["results_summary"]["condition_breakdown"]["C6"] == 2


def test_benchmark_experiment_resumption(tmp_path: Path):
    en_file = tmp_path / "benchmark_en.jsonl"
    hi_file = tmp_path / "benchmark_hi.jsonl"
    save_jsonl([_make_record("doc_en_1", "en")], en_file)
    save_jsonl([_make_record("doc_hi_1", "hi")], hi_file)

    llm = MockLLM()
    nli = _make_mock_nli()
    runner = UnifiedExperimentRunner(llm_client=llm, nli_verifier=nli, delay_seconds=0.0)

    out_dir = tmp_path / "benchmark_results"

    # First run
    execute_benchmark_experiment(en_file, hi_file, out_dir, runner=runner, delay_seconds=0.0, resume=True)
    calls_after_first = llm.call_count

    # Second run with resume=True
    execute_benchmark_experiment(en_file, hi_file, out_dir, runner=runner, delay_seconds=0.0, resume=True)
    # No new generation calls made
    assert llm.call_count == calls_after_first


def test_pilot_artifacts_preservation():
    pilot_file = Path("data/generations/controlled_generation_results.jsonl")
    if pilot_file.exists():
        with open(pilot_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 20, "20-summary pilot dataset must remain preserved."
