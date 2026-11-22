"""Configuration definitions for multilingual NLI verification."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NLIConfig:
    """Configuration for NLI model loading and inference.

    Attributes:
        model_id: Model repository identifier or hosted model name.
        provider: Inference provider ('local' for HuggingFace Transformers, 'cohere' for Cohere API).
        device: Device to run inference on ('cpu', 'mps', 'cuda'). Defaults to 'cpu'.
        batch_size: Default batch size for batched inference. Conservative default for CPU.
        max_length: Maximum sequence length (tokens). Inputs exceeding this are truncated.
        truncation_strategy: Truncation behavior ('only_first', 'longest_first', True).
            Defaults to 'only_first' so the premise (source article) is truncated while
            the hypothesis (summary statement) is preserved.
        cache_dir: Optional custom Hugging Face model cache directory.
        api_key: Optional API key for hosted NLI inference. If None, resolved from env.
        api_base_url: Optional base URL for hosted API inference.
        timeout: Network timeout in seconds for API inference.
        max_retries: Maximum retry attempts for transient network/rate errors.
    """

    model_id: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    model_revision: str = "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"
    provider: str = "local"
    device: str = "cpu"
    batch_size: int = 4
    max_length: int = 512
    truncation_strategy: str = "only_first"
    cache_dir: Optional[str] = None
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 8
