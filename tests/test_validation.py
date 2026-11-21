"""Unit tests for text cleaning, normalization, and record validation."""

import pytest
from factindic.schema import FactIndicRecord
from factindic.validation import clean_and_normalize_text, validate_record


def test_clean_and_normalize_text_basic():
    raw = "   \n\r  This is a   sentence with messy whitespace. \r\n\t  "
    cleaned = clean_and_normalize_text(raw)
    assert cleaned.startswith("This")
    assert cleaned.endswith("whitespace.")


def test_clean_and_normalize_text_unicode_nfc():
    # Test Latin accent composition under NFC: 'e' + combining acute accent -> 'é'
    decomposed_latin = "e\u0301"
    composed_latin = "\u00e9"
    assert clean_and_normalize_text(decomposed_latin) == composed_latin

    # Test Devanagari Unicode NFC canonical form: both precomposed and decomposed map to canonical NFC
    precomposed_devanagari = "\u0958"
    decomposed_devanagari = "क\u093C"
    assert clean_and_normalize_text(precomposed_devanagari) == clean_and_normalize_text(decomposed_devanagari)
    assert clean_and_normalize_text(decomposed_devanagari) == "\u0915\u093c"


def test_clean_and_normalize_text_none_and_empty():
    assert clean_and_normalize_text(None) == ""
    assert clean_and_normalize_text("") == ""
    assert clean_and_normalize_text("   \n\t  ") == ""


def test_validate_record_valid():
    rec = FactIndicRecord(
        example_id="rec_val_01",
        language="en",
        source_text="This is a valid source text of sufficient length.",
        reference_summary="This is a valid summary.",
    )
    is_valid, reason = validate_record(rec)
    assert is_valid is True
    assert reason is None


def test_validate_record_empty_source():
    rec = FactIndicRecord(
        example_id="rec_val_02",
        language="en",
        source_text="   \n\t  ",
        reference_summary="This is a valid summary.",
    )
    is_valid, reason = validate_record(rec)
    assert is_valid is False
    assert "source_text" in reason.lower()


def test_validate_record_empty_summary():
    rec = FactIndicRecord(
        example_id="rec_val_03",
        language="en",
        source_text="This is a valid source text of sufficient length.",
        reference_summary="",
    )
    is_valid, reason = validate_record(rec)
    assert is_valid is False
    assert "reference_summary" in reason.lower()


def test_validate_record_missing_id():
    rec = FactIndicRecord(
        example_id="  ",
        language="en",
        source_text="This is a valid source text of sufficient length.",
        reference_summary="This is a valid summary.",
    )
    is_valid, reason = validate_record(rec)
    assert is_valid is False
    assert "example_id" in reason.lower()


def test_validate_record_too_short():
    rec = FactIndicRecord(
        example_id="rec_val_05",
        language="en",
        source_text="Short",
        reference_summary="Valid summary.",
    )
    is_valid, reason = validate_record(rec)
    assert is_valid is False
    assert "too short" in reason.lower()
