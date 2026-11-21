"""Unit tests for the Full-Scale Benchmark Dataset Preparation Pipeline.

Tests:
    - Deterministic sampling with seed 42
    - English and Hindi separation (disjoint sets)
    - Duplicate prevention across IDs and contents
    - Schema validation and handling of empty/invalid records
    - Configurable sample sizes
    - Insufficient valid records handling without fabricating data
    - Provenance audit structure and correctness
    - JSONL serialization and roundtrip reading
    - Preservation of existing pilot artifacts (controlled_generation_results.jsonl, data/samples/)
"""

import json
from pathlib import Path
from typing import List
import pytest

from factindic.datasets.base import BaseDatasetLoader
from factindic.schema import FactIndicRecord
from factindic.serialization import load_jsonl
from scripts.prepare_benchmark import (
    generate_benchmark_audit,
    prepare_language_benchmark,
    run_benchmark_preparation,
)


class MockBenchmarkLoader(BaseDatasetLoader):
    """Deterministic mock loader returning controlled records for testing."""

    def __init__(self, records: List[FactIndicRecord]):
        self.records = records

    def load(self, split: str = "test", max_samples: int = None) -> List[FactIndicRecord]:
        return list(self.records)

    def stream(self, split: str = "test", max_samples: int = None):
        yield from self.records


def _make_test_record(
    record_id: str,
    language: str,
    source_text: str = None,
    reference_summary: str = None,
) -> FactIndicRecord:
    src = source_text if source_text is not None else f"A valid source article body for testing record {record_id}."
    ref = reference_summary if reference_summary is not None else f"A valid reference summary for record {record_id}."
    return FactIndicRecord(
        example_id=f"xlsum_{language}_{record_id}",
        language=language,
        source_text=src,
        reference_summary=ref,
        metadata={
            "raw_id": record_id,
            "title": f"Title for {record_id}",
            "split": "test",
            "source_dataset": "csebuetnlp/xlsum",
        },
    )


def test_deterministic_sampling_seed_42():
    records = [_make_test_record(f"doc_{i}", "en") for i in range(50)]
    loader = MockBenchmarkLoader(records)

    sampled1, meta1 = prepare_language_benchmark(
        loader=loader,
        language="en",
        sample_size=10,
        seed=42,
    )
    sampled2, meta2 = prepare_language_benchmark(
        loader=loader,
        language="en",
        sample_size=10,
        seed=42,
    )

    assert len(sampled1) == 10
    assert len(sampled2) == 10
    assert [r.example_id for r in sampled1] == [r.example_id for r in sampled2]
    assert meta1["record_ids"] == meta2["record_ids"]


def test_different_seeds_produce_different_orders():
    records = [_make_test_record(f"doc_{i}", "en") for i in range(50)]
    loader = MockBenchmarkLoader(records)

    sampled_42, _ = prepare_language_benchmark(loader=loader, language="en", sample_size=10, seed=42)
    sampled_99, _ = prepare_language_benchmark(loader=loader, language="en", sample_size=10, seed=99)

    assert [r.example_id for r in sampled_42] != [r.example_id for r in sampled_99]


def test_english_and_hindi_separation():
    en_records = [_make_test_record(f"en_{i}", "en") for i in range(15)]
    hi_records = [_make_test_record(f"hi_{i}", "hi") for i in range(15)]

    en_loader = MockBenchmarkLoader(en_records)
    hi_loader = MockBenchmarkLoader(hi_records)

    en_sampled, en_meta = prepare_language_benchmark(loader=en_loader, language="en", sample_size=5)
    hi_sampled, hi_meta = prepare_language_benchmark(loader=hi_loader, language="hi", sample_size=5)

    assert len(en_sampled) == 5
    assert len(hi_sampled) == 5
    assert all(r.language == "en" for r in en_sampled)
    assert all(r.language == "hi" for r in hi_sampled)

    en_ids = set(en_meta["record_ids"])
    hi_ids = set(hi_meta["record_ids"])
    assert en_ids.isdisjoint(hi_ids)


def test_duplicate_prevention():
    records = [
        _make_test_record("doc_dup_1", "en", source_text="Exact same content.", reference_summary="Same summary."),
        _make_test_record("doc_dup_1", "en", source_text="Exact same content.", reference_summary="Same summary."),  # Duplicate ID & content
        _make_test_record("doc_dup_2", "en", source_text="Exact same content.", reference_summary="Same summary."),  # Duplicate content
        _make_test_record("doc_unique_3", "en", source_text="Distinct unique content.", reference_summary="Distinct summary."),
    ]
    loader = MockBenchmarkLoader(records)

    sampled, meta = prepare_language_benchmark(
        loader=loader,
        language="en",
        sample_size=10,
    )

    # Exactly 2 distinct records should survive deduplication
    assert len(sampled) == 2
    assert meta["duplicate_count"] == 2
    ids = [r.example_id for r in sampled]
    assert len(ids) == len(set(ids))


def test_invalid_and_empty_records_filtered():
    records = [
        _make_test_record("doc_valid", "en"),
        _make_test_record("doc_empty_src", "en", source_text=""),  # Invalid: empty source
        _make_test_record("doc_empty_sum", "en", reference_summary=""),  # Invalid: empty summary
    ]
    loader = MockBenchmarkLoader(records)

    sampled, meta = prepare_language_benchmark(
        loader=loader,
        language="en",
        sample_size=5,
    )

    assert len(sampled) == 1
    assert sampled[0].example_id == "xlsum_en_doc_valid"
    assert meta["invalid_records_count"] == 2


def test_configurable_sample_sizes():
    records = [_make_test_record(f"doc_{i}", "en") for i in range(30)]
    loader = MockBenchmarkLoader(records)

    for target_size in (3, 7, 15, 25):
        sampled, meta = prepare_language_benchmark(
            loader=loader,
            language="en",
            sample_size=target_size,
        )
        assert len(sampled) == target_size
        assert meta["sample_size_obtained"] == target_size
        assert meta["sample_size_requested"] == target_size


def test_insufficient_valid_records_reports_actual_count():
    # Provide only 4 valid records when 20 are requested
    records = [_make_test_record(f"doc_{i}", "en") for i in range(4)]
    loader = MockBenchmarkLoader(records)

    sampled, meta = prepare_language_benchmark(
        loader=loader,
        language="en",
        sample_size=20,
    )

    assert len(sampled) == 4
    assert meta["sample_size_obtained"] == 4
    assert meta["sample_size_requested"] == 20
    # Confirms no records were fabricated
    assert [r.example_id for r in sampled] == [r.example_id for r in records]


def test_audit_report_generation(tmp_path: Path):
    en_meta = {
        "language": "en",
        "split": "test",
        "sample_size_requested": 100,
        "sample_size_obtained": 100,
        "record_ids": ["xlsum_en_1", "xlsum_en_2"],
    }
    hi_meta = {
        "language": "hi",
        "split": "test",
        "sample_size_requested": 100,
        "sample_size_obtained": 100,
        "record_ids": ["xlsum_hi_1", "xlsum_hi_2"],
    }

    audit_file = tmp_path / "benchmark_audit.json"
    audit = generate_benchmark_audit(
        en_meta=en_meta,
        hi_meta=hi_meta,
        seed=42,
        split="test",
        source_format="parquet",
        output_audit_path=audit_file,
    )

    assert audit["benchmark_name"] == "FactIndic-Full-Scale-Benchmark"
    assert audit["dataset_name"] == "csebuetnlp/xlsum"
    assert audit["sampling_seed"] == 42
    assert audit["split"] == "test"
    assert audit["licensing"]["source_license"] == "CC BY-NC-SA 4.0"
    assert audit["summary"]["total_requested"] == 200
    assert audit["summary"]["total_obtained"] == 200
    assert audit_file.exists()

    with open(audit_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["benchmark_name"] == audit["benchmark_name"]


def test_run_benchmark_preparation_fixture_mode(tmp_path: Path):
    audit = run_benchmark_preparation(
        en_size=3,
        hi_size=3,
        seed=42,
        output_dir=tmp_path,
        fixture_test_mode_only=True,
    )

    en_out = tmp_path / "benchmark_en.jsonl"
    hi_out = tmp_path / "benchmark_hi.jsonl"
    aud_out = tmp_path / "benchmark_audit.json"

    assert en_out.exists()
    assert hi_out.exists()
    assert aud_out.exists()

    en_records = load_jsonl(en_out)
    hi_records = load_jsonl(hi_out)

    assert len(en_records) == 3
    assert len(hi_records) == 3
    assert all(r.language == "en" for r in en_records)
    assert all(r.language == "hi" for r in hi_records)


def test_preservation_of_previous_pilot_artifacts():
    pilot_file = Path("data/generations/controlled_generation_results.jsonl")
    if pilot_file.exists():
        with open(pilot_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 20, "20-summary pilot dataset must remain preserved."
