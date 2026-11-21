"""Unit tests for controlled summary generation experiment, validation, and resumption."""

import json
from pathlib import Path
from typing import Dict, List
import pytest

from factindic.controlled_experiment import (
    CONTROLLED_GENERATION_PARAMS,
    CONTROLLED_MODEL_ID,
    CONTROLLED_PROVIDER,
    execute_controlled_experiment,
    validate_controlled_records,
)
from factindic.generation import GenerationRecord, load_generation_records
from factindic.models.base import BaseLLMClient, GenerationResult
from factindic.prompts.templates import PROMPT_TEMPLATE_VERSION, PromptStrategy
from factindic.schema import FactIndicRecord


class ControlledMockClient(BaseLLMClient):
    """Deterministic mock client tracking invocation counts for testing."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.call_count = 0
        self.provider = CONTROLLED_PROVIDER
        self.model_id = CONTROLLED_MODEL_ID

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_new_tokens: int = 256,
        top_p: float = 0.9,
    ) -> GenerationResult:
        self.call_count += 1
        if self.should_fail:
            raise ConnectionError("Mock network failure.")
        return GenerationResult(
            text=f"Mock summary output #{self.call_count} from {self.model_id}.",
            model_id=self.model_id,
            provider=self.provider,
            generation_parameters={
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "top_p": top_p,
            },
            latency_seconds=0.05,
        )


def _make_dummy_records(language: str, count: int = 5) -> List[FactIndicRecord]:
    lang_code = "en" if language == "english" else "hi"
    records = []
    for i in range(1, count + 1):
        raw_id = f"{lang_code}_doc_{i}"
        rec = FactIndicRecord(
            example_id=f"xlsum_{lang_code}_{raw_id}",
            language=lang_code,
            source_text=f"Source article text for {raw_id} with sufficient length for testing.",
            reference_summary=f"Reference summary for {raw_id}.",
            metadata={"raw_id": raw_id, "split": "test", "source_dataset": "csebuetnlp/xlsum"},
        )
        records.append(rec)
    return records


def test_validate_controlled_records_valid():
    """Verify validation passes when all 20 expected records are present and valid."""
    en_records = _make_dummy_records("english", 5)
    hi_records = _make_dummy_records("hindi", 5)
    expected_ids = {
        "english": [r.metadata["raw_id"] for r in en_records],
        "hindi": [r.metadata["raw_id"] for r in hi_records],
    }

    records = []
    for lang, r_list in [("en", en_records), ("hi", hi_records)]:
        for r in r_list:
            for strat in ("prompt_a", "prompt_b"):
                rec = GenerationRecord(
                    source_dataset="csebuetnlp/xlsum",
                    source_config=lang,
                    source_split="test",
                    source_record_id=r.metadata["raw_id"],
                    language=lang,
                    source_text=r.source_text,
                    reference_summary=r.reference_summary,
                    prompt_strategy=strat,
                    prompt_template_version=PROMPT_TEMPLATE_VERSION,
                    model_id=CONTROLLED_MODEL_ID,
                    provider=CONTROLLED_PROVIDER,
                    generation_parameters=CONTROLLED_GENERATION_PARAMS,
                    generated_summary=f"Valid summary for {r.metadata['raw_id']} under {strat}.",
                    generation_timestamp_utc="2026-11-21T00:00:00Z",
                    latency_seconds=0.1,
                    status="SUCCESS",
                )
                records.append(rec)

    report = validate_controlled_records(records, expected_ids)
    assert report["is_valid"] is True
    assert report["total_records"] == 20
    assert report["counts"]["total_success"] == 20
    assert report["counts"]["english"]["prompt_a_success"] == 5
    assert report["counts"]["english"]["prompt_b_success"] == 5
    assert report["counts"]["hindi"]["prompt_a_success"] == 5
    assert report["counts"]["hindi"]["prompt_b_success"] == 5
    assert len(report["errors"]) == 0


def test_validate_controlled_records_detects_empty_summary():
    """Verify validation detects when a record contains an empty summary."""
    en_records = _make_dummy_records("english", 1)
    expected_ids = {"english": ["en_doc_1"], "hindi": []}

    rec = GenerationRecord(
        source_dataset="csebuetnlp/xlsum",
        source_config="en",
        source_split="test",
        source_record_id="en_doc_1",
        language="en",
        source_text="Some text",
        reference_summary="Some ref",
        prompt_strategy="prompt_a",
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        model_id=CONTROLLED_MODEL_ID,
        provider=CONTROLLED_PROVIDER,
        generation_parameters=CONTROLLED_GENERATION_PARAMS,
        generated_summary="   ",  # Empty whitespace
        generation_timestamp_utc="2026-11-21T00:00:00Z",
        status="SUCCESS",
    )

    report = validate_controlled_records([rec], expected_ids)
    assert report["is_valid"] is False
    assert any("empty generated_summary" in e for e in report["errors"])


def test_execute_controlled_experiment_and_resumption(tmp_path: Path):
    """Verify end-to-end controlled generation and idempotence upon resumption."""
    en_records = _make_dummy_records("english", 5)
    hi_records = _make_dummy_records("hindi", 5)

    results_file = tmp_path / "controlled_results.jsonl"
    audit_file = tmp_path / "controlled_audit.json"

    client = ControlledMockClient(should_fail=False)

    # First run: should execute all 20 calls
    records, audit = execute_controlled_experiment(
        client=client,
        en_records=en_records,
        hi_records=hi_records,
        results_path=results_file,
        audit_path=audit_file,
        delay_seconds=0.0,
        max_retries=1,
    )

    assert len(records) == 20
    assert client.call_count == 20
    assert audit["validation_passed"] is True
    assert audit["expected_counts"]["total_summaries"] == 20
    assert audit["actual_counts"]["total_success"] == 20
    assert results_file.exists()
    assert audit_file.exists()

    # Second run (resumption): should detect all 20 already completed and make 0 new API calls!
    records_resumed, audit_resumed = execute_controlled_experiment(
        client=client,
        en_records=en_records,
        hi_records=hi_records,
        results_path=results_file,
        audit_path=audit_file,
        delay_seconds=0.0,
        max_retries=1,
    )

    assert len(records_resumed) == 20
    assert client.call_count == 20  # Still 20, no new calls made
    assert audit_resumed["validation_passed"] is True
