"""Result and label schemas for Natural Language Inference (NLI) verification."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class NLILabel(str, Enum):
    """Canonical NLI class labels."""

    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


@dataclass
class NLIResult:
    """Structured result returned by an NLI verification inference.

    Attributes:
        premise: The source article or evidence text serving as the premise.
        hypothesis: The summary statement or claim serving as the hypothesis.
        language: Language code for the input pair ('en', 'hi', etc.).
        predicted_label: The class label with the highest predicted probability.
        entailment_prob: Normalized probability that the hypothesis is entailed by premise.
        neutral_prob: Normalized probability that the hypothesis is neutral/unsupported.
        contradiction_prob: Normalized probability that the hypothesis is contradicted by premise.
        probabilities: Full dictionary mapping each label to its normalized probability.
        model_id: Hugging Face model identifier used for inference.
        latency_ms: Inference execution time in milliseconds.
    """

    premise: str
    hypothesis: str
    language: str
    predicted_label: str
    entailment_prob: float
    neutral_prob: float
    contradiction_prob: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    model_id: str = ""
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.probabilities:
            self.probabilities = {
                NLILabel.CONTRADICTION.value: self.contradiction_prob,
                NLILabel.NEUTRAL.value: self.neutral_prob,
                NLILabel.ENTAILMENT.value: self.entailment_prob,
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert result object to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NLIResult":
        """Construct NLIResult from a dictionary representation."""
        return cls(
            premise=str(data.get("premise", "")),
            hypothesis=str(data.get("hypothesis", "")),
            language=str(data.get("language", "")),
            predicted_label=str(data.get("predicted_label", "")),
            entailment_prob=float(data.get("entailment_prob", 0.0)),
            neutral_prob=float(data.get("neutral_prob", 0.0)),
            contradiction_prob=float(data.get("contradiction_prob", 0.0)),
            probabilities=dict(data.get("probabilities", {})),
            model_id=str(data.get("model_id", "")),
            latency_ms=float(data.get("latency_ms", 0.0)),
        )
