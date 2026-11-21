"""Configuration definitions for multilingual NLI verification."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NLIConfig:
    """Configuration for NLI model loading and inference.

    Attributes:
        model_id: Hugging Face model repository identifier.
        device: Device to run inference on ('cpu', 'mps', 'cuda'). Defaults to 'cpu'.
        batch_size: Default batch size for batched inference. Conservative default for CPU.
        max_length: Maximum sequence length (tokens). Inputs exceeding this are truncated.
        truncation_strategy: Truncation behavior ('only_first', 'longest_first', True).
            Defaults to 'only_first' so the premise (source article) is truncated while
            the hypothesis (summary statement) is preserved.
        cache_dir: Optional custom Hugging Face model cache directory.
    """

    model_id: str = "joeddav/xlm-roberta-large-xnli"
    device: str = "cpu"
    batch_size: int = 4
    max_length: int = 512
    truncation_strategy: str = "only_first"
    cache_dir: Optional[str] = None
