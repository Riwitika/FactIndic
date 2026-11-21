"""Tests for FactIndic Streamlit UI helpers."""

import json
from pathlib import Path
from factindic.schema import FactIndicRecord
from factindic.ui_helpers import (
    TARGET_RECORD_IDS,
    TARGET_TITLES,
    fetch_live_target_records,
    get_pipeline_stages,
    get_project_audits,
    load_audit_file,
    load_demo_generations,
)


def test_target_constants():
    """Verify target record IDs and titles for English and Hindi."""
    assert "english" in TARGET_RECORD_IDS
    assert "hindi" in TARGET_RECORD_IDS
    assert "world-us-canada-38697908" in TARGET_RECORD_IDS["english"]
    assert "uk-wales-politics-48586353" in TARGET_RECORD_IDS["english"]
    assert "081027_sadhvi_test_pp" in TARGET_RECORD_IDS["hindi"]
    assert "070929_pakistan_protest" in TARGET_RECORD_IDS["hindi"]
    assert len(TARGET_TITLES) >= 4


def test_load_audit_file(tmp_path: Path):
    """Test loading valid JSON and handling nonexistent files."""
    test_json = tmp_path / "test_audit.json"
    test_json.write_text(json.dumps({"status": "verified", "count": 4}), encoding="utf-8")

    loaded = load_audit_file(test_json)
    assert loaded is not None
    assert loaded["status"] == "verified"
    assert loaded["count"] == 4

    missing = load_audit_file(tmp_path / "nonexistent.json")
    assert missing is None


def test_get_project_audits():
    """Test reading project audits dictionary."""
    audits = get_project_audits()
    assert "live_xlsum" in audits
    assert "demo_generation" in audits
    assert "smoke_generation" in audits
    # At least demo_generation or smoke_generation should be present in data/
    if audits["demo_generation"] is not None:
        assert audits["demo_generation"].get("final_research_model") == "UNDECIDED"
        assert audits["demo_generation"].get("experiment_type") == "demo_only"


def test_load_demo_generations_empty():
    """Test loading demo generations returns list without failing."""
    records = load_demo_generations()
    assert isinstance(records, list)


def test_get_pipeline_stages():
    """Verify research pipeline stage metadata."""
    stages = get_pipeline_stages()
    assert len(stages) == 7
    stage_titles = [s["title"] for s in stages]
    assert any("Whole-Doc NLI" in t for t in stage_titles)
    assert any("Claim-Level NLI" in t for t in stage_titles)
    assert any("Hybrid" in t for t in stage_titles)
    assert any("Human Evaluation" in t for t in stage_titles)

    # Check planned statuses
    planned_stages = [s for s in stages if "Planned" in s["status"]]
    assert len(planned_stages) >= 4


def test_factindic_record_attribute_contract():
    """Verify FactIndicRecord attributes match app.py Dataset Explorer expectations."""
    from factindic.schema import FactIndicRecord

    rec = FactIndicRecord(
        example_id="xlsum_en_123",
        language="en",
        source_text="This is a test source text for the article.",
        reference_summary="This is the reference summary.",
        metadata={"raw_id": "123", "title": "Test Title", "url": "https://bbc.com", "split": "test"},
    )

    # Valid schema attributes used in app.py
    assert rec.source_text == "This is a test source text for the article."
    assert rec.reference_summary == "This is the reference summary."
    assert rec.metadata.get("split") == "test"
    assert rec.metadata.get("raw_id") == "123"

    # Negative contract checks: Ensure schema does not expose nonexistent fields that cause AttributeError
    assert not hasattr(rec, "split")
    assert not hasattr(rec, "summary_text")

    # Simulate UI extraction logic from app.py
    ui_record_id = rec.metadata.get("raw_id", rec.example_id)
    ui_split = rec.metadata.get("split", "test")
    ui_source_text = rec.source_text
    ui_ref_summary = rec.reference_summary

    assert ui_record_id == "123"
    assert ui_split == "test"
    assert len(ui_source_text) > 0
    assert len(ui_ref_summary) > 0


def test_fetch_live_target_records_scans_beyond_25():
    """Verify fetch_live_target_records finds target records situated beyond index 25."""
    from unittest.mock import patch

    # Generate 40 dummy records, with target record placed at position 35
    target_id = TARGET_RECORD_IDS["english"][0]

    def mock_stream_raw(split="test", max_samples=None):
        count = 0
        for i in range(50):
            raw_id = target_id if i == 35 else f"dummy_{i}"
            raw = {
                "id": raw_id,
                "title": f"Title {i}",
                "text": f"Article text for record {i} of sufficient length.",
                "summary": f"Summary for record {i}.",
            }
            rec = FactIndicRecord(
                example_id=f"xlsum_en_{raw_id}",
                language="en",
                source_text=raw["text"],
                reference_summary=raw["summary"],
                metadata={"raw_id": raw_id, "split": split, "title": raw["title"]},
            )
            yield raw, rec
            count += 1
            if max_samples is not None and count >= max_samples:
                break

    with patch("factindic.ui_helpers.XLSumLoader") as MockLoader:
        mock_instance = MockLoader.return_value
        mock_instance.stream_raw.side_effect = mock_stream_raw

        # Default scan (max_scan_samples=None) must reach position 35 and find target
        records, err = fetch_live_target_records("english")
        assert err is None
        found_ids = [r.metadata.get("raw_id") for r in records]
        assert target_id in found_ids

        # If capped at 25, it should miss the record at 35
        records_capped, _ = fetch_live_target_records("english", max_scan_samples=25)
        found_capped_ids = [r.metadata.get("raw_id") for r in records_capped]
        assert target_id not in found_capped_ids

