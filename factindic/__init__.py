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
from factindic.cache import ResponseCache, CacheStats, compute_cache_key
from factindic.benchmark_manifest import (
    compute_file_sha256,
    generate_benchmark_manifest,
    validate_benchmark_manifest,
)
from factindic.preflight import estimate_api_calls, run_preflight
from factindic.exceptions import (
    FactIndicError,
    AuthenticationError,
    RateLimitError,
    QuotaExhaustedError,
    InferenceTimeoutError,
    MalformedResponseError,
)
from factindic.experiment_runner import (
    ConditionID,
    ExperimentRecord,
    UnifiedExperimentRunner,
    load_experiment_records,
    save_experiment_records,
)

__version__ = "0.1.0"

__all__ = [
    "AuthenticationError",
    "CacheStats",
    "ConditionID",
    "DatasetStatistics",
    "ExperimentRecord",
    "FactIndicError",
    "FactIndicRecord",
    "InferenceTimeoutError",
    "MalformedResponseError",
    "QuotaExhaustedError",
    "RateLimitError",
    "ResponseCache",
    "UnifiedExperimentRunner",
    "clean_and_normalize_text",
    "compute_cache_key",
    "compute_dataset_statistics",
    "compute_file_sha256",
    "estimate_api_calls",
    "generate_benchmark_manifest",
    "load_experiment_records",
    "load_jsonl",
    "run_preflight",
    "sample_records",
    "save_experiment_records",
    "save_jsonl",
    "validate_benchmark_manifest",
    "validate_dataset",
    "validate_record",
]
