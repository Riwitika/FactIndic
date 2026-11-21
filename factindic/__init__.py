"""FactIndic: Factuality-Aware Summarization for Indic and Code-Mixed Languages.

Phase 1: Dataset Preparation and Validation Pipeline.
"""

from factindic.schema import FactIndicRecord, DatasetStatistics
from factindic.validation import (
    validate_record,
    clean_and_normalize_text,
    validate_dataset,
    compute_dataset_statistics,
)
from factindic.sampling import sample_records
from factindic.serialization import load_jsonl, save_jsonl

__version__ = "0.1.0"

__all__ = [
    "FactIndicRecord",
    "DatasetStatistics",
    "validate_record",
    "clean_and_normalize_text",
    "validate_dataset",
    "compute_dataset_statistics",
    "sample_records",
    "load_jsonl",
    "save_jsonl",
]
