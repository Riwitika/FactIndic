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
from factindic.verification.claim_extractor import (
    AtomicClaimExtractor,
    ExtractedClaim,
    load_extracted_claims,
    save_extracted_claims,
)
from factindic.verification.rule_claim_extractor import (
    RuleBasedClaimExtractor,
    clean_preamble,
    decompose_english_sentence,
    decompose_hindi_sentence,
    is_meta_statement,
)
from factindic.verification.claim_verifier import (
    ClaimVerificationResult,
    HybridClaimVerifier,
    load_claim_verification_results,
    save_claim_verification_results,
)
from factindic.verification.evidence_retriever import (
    BM25EvidenceRetriever,
    ClaimEvidenceResult,
    EvidencePassage,
    load_evidence_results,
    save_evidence_results,
    segment_article_into_passages,
    tokenize_text,
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
    "AtomicClaimExtractor",
    "BM25EvidenceRetriever",
    "ClaimEvidenceResult",
    "ClaimVerificationResult",
    "DocumentVerificationResult",
    "EvidencePassage",
    "ExtractedClaim",
    "HybridClaimVerifier",
    "NLIConfig",
    "NLILabel",
    "NLIResult",
    "RuleBasedClaimExtractor",
    "SUPPORTED_VERIFICATION_LANGUAGES",
    "SentenceLevelVerifier",
    "SentenceVerificationResult",
    "VerificationDependencyError",
    "VerificationModelLoadError",
    "WholeDocumentVerifier",
    "XNLIVerifier",
    "load_claim_verification_results",
    "load_document_verification_results",
    "load_evidence_results",
    "load_extracted_claims",
    "load_sentence_verification_results",
    "save_claim_verification_results",
    "save_document_verification_results",
    "save_evidence_results",
    "save_extracted_claims",
    "save_sentence_verification_results",
    "segment_article_into_passages",
    "split_into_sentences",
    "tokenize_text",
]
