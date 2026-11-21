"""Unit tests for dataset statistics computation."""

import pytest
from factindic.schema import FactIndicRecord
from factindic.validation import validate_dataset, compute_dataset_statistics


def test_statistics_breakdown():
    records = [
        # 2 Valid English records
        FactIndicRecord("en_1", "en", "First valid English source text of sufficient length.", "Summary one."),
        FactIndicRecord("en_2", "en", "Second valid English source text of sufficient length.", "Summary two."),
        # 1 Valid Hindi record
        FactIndicRecord("hi_1", "hi", "यह पहला वैध हिंदी स्रोत पाठ है जिसकी लंबाई पर्याप्त है।", "संक्षिप्त सारांश।"),
        # 1 Invalid empty source
        FactIndicRecord("inv_1", "en", "   ", "Summary."),
        # 1 Invalid short summary
        FactIndicRecord("inv_2", "hi", "वैध स्रोत पाठ लेकिन छोटा सारांश।", "कम"),
        # 1 Duplicate of en_1
        FactIndicRecord("en_1", "en", "First valid English source text of sufficient length.", "Summary one."),
    ]

    valid, invalid, stats = validate_dataset(records, detect_duplicates=True)

    assert stats.total_loaded == 6
    assert stats.valid_count == 3
    assert stats.invalid_count == 3
    assert stats.duplicate_count == 1
    assert stats.language_distribution == {"en": 2, "hi": 1}
    assert stats.average_source_length_chars > 0
    assert stats.average_summary_length_chars > 0
    assert isinstance(stats.to_dict(), dict)
