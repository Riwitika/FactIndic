"""Schema definitions for FactIndic records and dataset statistics."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class FactIndicRecord:
    """Standardized schema for FactIndic benchmark records.

    Attributes:
        example_id: Globally unique identifier for this record (e.g., 'xlsum_en_12345').
        language: Language code / name (e.g., 'en', 'hi', 'english', 'hindi').
        source_text: Cleaned, non-empty full source article text.
        reference_summary: Cleaned, non-empty reference summary.
        metadata: Optional dictionary containing provenance, URLs, titles, split, etc.
    """

    example_id: str
    language: str
    source_text: str
    reference_summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to Python dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FactIndicRecord":
        """Construct record from dictionary with validation."""
        return cls(
            example_id=str(data.get("example_id", "")),
            language=str(data.get("language", "")),
            source_text=str(data.get("source_text", "")),
            reference_summary=str(data.get("reference_summary", "")),
            metadata=dict(data.get("metadata", {})) if data.get("metadata") is not None else {},
        )


@dataclass
class DatasetStatistics:
    """Statistical summary of dataset loading and validation results."""

    total_loaded: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    duplicate_count: int = 0
    language_distribution: Dict[str, int] = field(default_factory=dict)
    invalid_reasons: Dict[str, int] = field(default_factory=dict)
    average_source_length_chars: float = 0.0
    average_summary_length_chars: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return asdict(self)
