"""Unit tests for UTF-8 JSONL serialization and Unicode preservation."""

import json
import pytest
from pathlib import Path
from factindic.schema import FactIndicRecord
from factindic.serialization import save_jsonl, load_jsonl


def test_jsonl_serialization_and_roundtrip(tmp_path: Path):
    target = tmp_path / "test_records.jsonl"
    records = [
        FactIndicRecord(
            example_id="rec_en_01",
            language="en",
            source_text="English source article text.",
            reference_summary="English reference summary.",
            metadata={"split": "test"},
        ),
        FactIndicRecord(
            example_id="rec_hi_01",
            language="hi",
            source_text="भारतीय अंतरिक्ष अनुसंधान संगठन (इसरो) ने चंद्रयान-3 प्रक्षेपित किया।",
            reference_summary="इसरो ने चंद्रयान-3 का सफल प्रक्षेपण किया।",
            metadata={"split": "test"},
        ),
    ]

    written = save_jsonl(records, target, overwrite=True, ensure_ascii=False)
    assert written == 2
    assert target.exists()

    loaded = load_jsonl(target)
    assert len(loaded) == 2
    assert loaded[0].example_id == "rec_en_01"
    assert loaded[1].example_id == "rec_hi_01"
    assert "चंद्रयान-3" in loaded[1].source_text


def test_unicode_preservation_native_devanagari(tmp_path: Path):
    target = tmp_path / "unicode_check.jsonl"
    hindi_text = "हिंदी भाषा में तथ्यपरक सारांश की जांच।"
    hindi_summary = "तथ्यपरक सारांश।"
    records = [
        FactIndicRecord(
            example_id="rec_unicode_01",
            language="hi",
            source_text=hindi_text,
            reference_summary=hindi_summary,
        )
    ]

    save_jsonl(records, target, overwrite=True, ensure_ascii=False)

    # Read raw bytes to verify native UTF-8 encoding (not \\u09xx escape sequences)
    raw_content = target.read_text(encoding="utf-8")
    assert "\\u09" not in raw_content
    assert "हिंदी" in raw_content
    assert "तथ्यपरक" in raw_content


def test_overwrite_protection(tmp_path: Path):
    target = tmp_path / "protected.jsonl"
    records = [
        FactIndicRecord(
            example_id="rec_prot",
            language="en",
            source_text="Some source text.",
            reference_summary="Some summary.",
        )
    ]

    save_jsonl(records, target, overwrite=True)
    assert target.exists()

    # Attempting to save again with overwrite=False must raise FileExistsError
    with pytest.raises(FileExistsError):
        save_jsonl(records, target, overwrite=False)
