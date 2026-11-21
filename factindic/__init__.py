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
from factindic.experiment_runner import (
    ConditionID,
    ExperimentRecord,
    UnifiedExperimentRunner,
    load_experiment_records,
    save_experiment_records,
)

__version__ = "0.1.0"

__all__ = [
    "ConditionID",
    "DatasetStatistics",
    "ExperimentRecord",
    "FactIndicRecord",
    "UnifiedExperimentRunner",
    "clean_and_normalize_text",
    "compute_dataset_statistics",
    "load_experiment_records",
    "load_jsonl",
    "sample_records",
    "save_experiment_records",
    "save_jsonl",
    "validate_dataset",
    "validate_record",
]
