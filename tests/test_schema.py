"""Unit tests for FactIndic record schema and data structures."""

import pytest
from factindic.schema import FactIndicRecord, DatasetStatistics


def test_factindic_record_valid_creation():
    rec = FactIndicRecord(
        example_id="xlsum_en_001",
        language="en",
        source_text="This is a full source article explaining an event in detail.",
        reference_summary="This is the concise summary.",
        metadata={"title": "Test Title", "split": "test"},
    )
    assert rec.example_id == "xlsum_en_001"
    assert rec.language == "en"
    assert "source article" in rec.source_text
    assert "concise summary" in rec.reference_summary
    assert rec.metadata["title"] == "Test Title"


def test_factindic_record_to_dict_and_from_dict():
    original = FactIndicRecord(
        example_id="xlsum_hi_002",
        language="hi",
        source_text="भारतीय अंतरिक्ष अनुसंधान संगठन ने नया परीक्षण किया।",
        reference_summary="इसरो ने परीक्षण किया।",
        metadata={"topic": "space"},
    )
    d = original.to_dict()
    assert isinstance(d, dict)
    assert d["example_id"] == "xlsum_hi_002"
    assert d["language"] == "hi"

    reconstructed = FactIndicRecord.from_dict(d)
    assert reconstructed.example_id == original.example_id
    assert reconstructed.source_text == original.source_text
    assert reconstructed.metadata == original.metadata


def test_factindic_record_from_dict_defaults():
    # Test handling of missing metadata key
    raw = {
        "example_id": "rec_003",
        "language": "en",
        "source_text": "Sample text",
        "reference_summary": "Sample summary",
    }
    rec = FactIndicRecord.from_dict(raw)
    assert rec.metadata == {}
