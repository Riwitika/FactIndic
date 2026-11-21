"""FactIndic verification module.

Provides Natural Language Inference (NLI) based factuality evaluation adapters
and configuration for multilingual summarization verification.
"""

from factindic.verification.config import NLIConfig
from factindic.verification.nli import (
    SUPPORTED_VERIFICATION_LANGUAGES,
    VerificationDependencyError,
    VerificationModelLoadError,
    XNLIVerifier,
)
from factindic.verification.result import NLILabel, NLIResult

__all__ = [
    "NLIConfig",
    "NLILabel",
    "NLIResult",
    "SUPPORTED_VERIFICATION_LANGUAGES",
    "VerificationDependencyError",
    "VerificationModelLoadError",
    "XNLIVerifier",
]
