"""Abstract base interface for LLM generation clients."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class GenerationResult:
    """Standardized result returned by an LLM generation client."""

    text: str
    model_id: str
    provider: str
    generation_parameters: Dict[str, Any]
    latency_seconds: float
    raw_response: Optional[Dict[str, Any]] = None


class BaseLLMClient(ABC):
    """Abstract interface for LLM inference clients."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_new_tokens: int = 256,
        top_p: float = 0.9,
    ) -> GenerationResult:
        """Generate text completion from a prompt.

        Args:
            prompt: Text prompt string.
            temperature: Sampling temperature (must be consistent across conditions).
            max_new_tokens: Maximum output length in tokens.
            top_p: Nucleus sampling parameter.

        Returns:
            GenerationResult dataclass.
        """
        pass
