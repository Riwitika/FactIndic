"""FactIndic verification module.

Provides Natural Language Inference (NLI) based factuality evaluation adapters
and configuration for multilingual summarization verification.
"""

from factindic.verification.config import NLIConfig
from factindic.verification.document_verifier import (
    DocumentVerificationResult,
    WholeDocumentVerifier,
    load_document_verification_results,
    save_document_verification_results,
)
from factindic.verification.nli import (
    SUPPORTED_VERIFICATION_LANGUAGES,
    VerificationDependencyError,
    VerificationModelLoadError,
    XNLIVerifier,
)
from factindic.verification.result import NLILabel, NLIResult
from factindic.verification.sentence_verifier import (
    SentenceLevelVerifier,
    SentenceVerificationResult,
    load_sentence_verification_results,
    save_sentence_verification_results,
    split_into_sentences,
)

__all__ = [
    "DocumentVerificationResult",
    "NLIConfig",
    "NLILabel",
    "NLIResult",
    "SUPPORTED_VERIFICATION_LANGUAGES",
    "SentenceLevelVerifier",
    "SentenceVerificationResult",
    "VerificationDependencyError",
    "VerificationModelLoadError",
    "WholeDocumentVerifier",
    "XNLIVerifier",
    "load_document_verification_results",
    "load_sentence_verification_results",
    "save_document_verification_results",
    "save_sentence_verification_results",
    "split_into_sentences",
]
