"""Unit tests for duplicate detection and handling."""

import pytest
from factindic.schema import FactIndicRecord
from factindic.validation import validate_dataset


def test_duplicate_id_detection():
    records = [
        FactIndicRecord(
            example_id="dup_id_01",
            language="en",
            source_text="First unique source article discussing global economy.",
            reference_summary="First unique summary.",
        ),
        FactIndicRecord(
            example_id="dup_id_01",  # Same ID, different text
            language="en",
            source_text="Second distinct source article discussing scientific discoveries.",
            reference_summary="Second distinct summary.",
        ),
    ]
    valid, invalid, stats = validate_dataset(records, detect_duplicates=True)
    assert len(valid) == 1
    assert len(invalid) == 1
    assert stats.duplicate_count == 1
    assert "duplicate example_id" in invalid[0][1].lower()


def test_duplicate_content_detection():
    records = [
        FactIndicRecord(
            example_id="rec_content_01",
            language="en",
            source_text="Identical source article repeated multiple times.",
            reference_summary="Identical summary text.",
        ),
        FactIndicRecord(
            example_id="rec_content_02",  # Different ID, identical content
            language="en",
            source_text="Identical source article repeated multiple times.",
            reference_summary="Identical summary text.",
        ),
    ]
    valid, invalid, stats = validate_dataset(records, detect_duplicates=True)
    assert len(valid) == 1
    assert len(invalid) == 1
    assert stats.duplicate_count == 1
    assert "duplicate content" in invalid[0][1].lower()


def test_distinct_articles_preserved():
    # Distinct articles sharing metadata should NOT be discarded
    records = [
        FactIndicRecord(
            example_id="article_01",
            language="en",
            source_text="The central bank increased interest rates by 25 basis points.",
            reference_summary="Interest rates were raised by 25 basis points.",
            metadata={"source": "BBC News", "category": "Economy"},
        ),
        FactIndicRecord(
            example_id="article_02",
            language="en",
            source_text="The stock market showed moderate gains following the rate decision.",
            reference_summary="Stock indices rose moderately after the monetary announcement.",
            metadata={"source": "BBC News", "category": "Economy"},
        ),
    ]
    valid, invalid, stats = validate_dataset(records, detect_duplicates=True)
    assert len(valid) == 2
    assert len(invalid) == 0
    assert stats.duplicate_count == 0
