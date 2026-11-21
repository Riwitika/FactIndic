"""Multilingual Natural Language Inference (NLI) verification adapter.

Uses joeddav/xlm-roberta-large-xnli to evaluate premise-hypothesis entailment for
factual consistency evaluation across English and Hindi summary sentences.

Note on Factuality Assessment:
    Entailment probability serves as an empirical heuristic proxy for factual
    consistency with source evidence. It should NOT be treated as a definitive
    or ground-truth measure of real-world factual correctness, as models are subject
    to premise-length biases, lack of real-world knowledge, and linguistic domain shifts.
"""

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from factindic.verification.config import NLIConfig
from factindic.verification.result import NLILabel, NLIResult

SUPPORTED_VERIFICATION_LANGUAGES: Dict[str, str] = {
    "en": "en",
    "english": "en",
    "hi": "hi",
    "hindi": "hi",
}


class VerificationDependencyError(ImportError):
    """Raised when required dependencies (torch, transformers) are not installed."""

    pass


class VerificationModelLoadError(RuntimeError):
    """Raised when the NLI model or tokenizer fails to load."""

    pass


def _softmax(logits: Sequence[float]) -> List[float]:
    """Compute numerically stable softmax over a 1D sequence of float logits."""
    max_logit = max(logits)
    exp_vals = [math.exp(v - max_logit) for v in logits]
    sum_exp = sum(exp_vals)
    if sum_exp == 0.0:
        return [1.0 / len(logits)] * len(logits)
    return [v / sum_exp for v in exp_vals]


class XNLIVerifier:
    """Multilingual NLI verification adapter supporting lazy loading and CPU inference.

    Features:
        - Lazy loading: Neither model nor tokenizer is loaded until explicitly requested.
        - Dynamic label mapping: Resolves label indices from model.config.id2label/label2id.
        - Controlled premise truncation: Defaults to 'only_first' truncation to retain hypotheses.
        - Resource-conscious: CPU-only execution and explicit unload_model() for memory release.
    """

    def __init__(self, config: Optional[NLIConfig] = None) -> None:
        """Initialize the XNLIVerifier with configuration.

        Args:
            config: Optional NLIConfig instance. Uses default CPU settings if omitted.
        """
        self.config = config or NLIConfig()
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._label_indices: Optional[Dict[str, int]] = None

    @property
    def is_loaded(self) -> bool:
        """Check whether the model and tokenizer are currently loaded into memory."""
        return self._model is not None and self._tokenizer is not None

    def load_model(self) -> None:
        """Explicitly load the tokenizer and model into memory.

        Raises:
            VerificationDependencyError: If torch or transformers is not installed.
            VerificationModelLoadError: If loading from Hugging Face fails.
        """
        if self.is_loaded:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise VerificationDependencyError(
                "NLI verification requires 'transformers' and 'torch'. "
                "Install them via: pip install transformers torch sentencepiece"
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                cache_dir=self.config.cache_dir,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_id,
                cache_dir=self.config.cache_dir,
            )
            if self.config.device == "cpu":
                model = model.to("cpu")
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            self._label_indices = self._resolve_label_indices(model.config)

        except Exception as exc:
            self._tokenizer = None
            self._model = None
            self._label_indices = None
            raise VerificationModelLoadError(
                f"Failed to load NLI model '{self.config.model_id}': {exc}"
            ) from exc

    def unload_model(self) -> None:
        """Unload model and tokenizer from memory and trigger garbage collection."""
        self._model = None
        self._tokenizer = None
        self._label_indices = None
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _resolve_label_indices(self, model_config: Any) -> Dict[str, int]:
        """Resolve indices for entailment, neutral, and contradiction from model config.

        Reads model.config.id2label or model.config.label2id dynamically rather
        than assuming fixed indices.
        """
        id2label = getattr(model_config, "id2label", None)
        label2id = getattr(model_config, "label2id", None)

        canonical_map: Dict[str, int] = {}

        if id2label:
            for idx_raw, label_raw in id2label.items():
                label_clean = str(label_raw).strip().lower()
                idx = int(idx_raw)
                if "entail" in label_clean:
                    canonical_map[NLILabel.ENTAILMENT.value] = idx
                elif "contra" in label_clean:
                    canonical_map[NLILabel.CONTRADICTION.value] = idx
                elif "neutral" in label_clean:
                    canonical_map[NLILabel.NEUTRAL.value] = idx

        elif label2id:
            for label_raw, idx_raw in label2id.items():
                label_clean = str(label_raw).strip().lower()
                idx = int(idx_raw)
                if "entail" in label_clean:
                    canonical_map[NLILabel.ENTAILMENT.value] = idx
                elif "contra" in label_clean:
                    canonical_map[NLILabel.CONTRADICTION.value] = idx
                elif "neutral" in label_clean:
                    canonical_map[NLILabel.NEUTRAL.value] = idx

        # Default fallback for joeddav/xlm-roberta-large-xnli: 0=contradiction, 1=neutral, 2=entailment
        if NLILabel.CONTRADICTION.value not in canonical_map:
            canonical_map[NLILabel.CONTRADICTION.value] = 0
        if NLILabel.NEUTRAL.value not in canonical_map:
            canonical_map[NLILabel.NEUTRAL.value] = 1
        if NLILabel.ENTAILMENT.value not in canonical_map:
            canonical_map[NLILabel.ENTAILMENT.value] = 2

        return canonical_map

    def _validate_language(self, language: str) -> str:
        """Validate and return canonical 2-letter language code.

        Raises:
            ValueError: If the language is not supported for verification.
        """
        lang_clean = str(language).strip().lower()
        if lang_clean not in SUPPORTED_VERIFICATION_LANGUAGES:
            supported = ", ".join(sorted(set(SUPPORTED_VERIFICATION_LANGUAGES.keys())))
            raise ValueError(
                f"Unsupported verification language '{language}'. Supported languages: {supported}"
            )
        return SUPPORTED_VERIFICATION_LANGUAGES[lang_clean]

    def _validate_input_text(self, text: str, field_name: str) -> str:
        """Validate that input text is non-empty string.

        Raises:
            ValueError: If text is empty or only whitespace.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")
        return text.strip()

    def verify_premise_hypothesis(
        self,
        premise: str,
        hypothesis: str,
        language: str = "en",
    ) -> NLIResult:
        """Evaluate an individual premise-hypothesis pair.

        Args:
            premise: Source article or evidence passage.
            hypothesis: Summary statement or claim to verify.
            language: Language of the input pair ('en', 'hi', etc.).

        Returns:
            Structured NLIResult containing probabilities and predicted label.
        """
        results = self.verify_batch(
            pairs=[(premise, hypothesis)],
            language=language,
            batch_size=1,
        )
        return results[0]

    def verify_batch(
        self,
        pairs: Sequence[Tuple[str, str]],
        language: str = "en",
        batch_size: Optional[int] = None,
    ) -> List[NLIResult]:
        """Evaluate a batch of premise-hypothesis pairs.

        Args:
            pairs: Sequence of (premise, hypothesis) string tuples.
            language: Language code applicable to the pairs ('en' or 'hi').
            batch_size: Optional batch size override. Uses config.batch_size if omitted.

        Returns:
            List of NLIResult instances in the same order as input pairs.
        """
        if not pairs:
            return []

        canon_lang = self._validate_language(language)

        # Validate all inputs before loading model or starting inference
        validated_pairs: List[Tuple[str, str]] = []
        for i, pair in enumerate(pairs):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(
                    f"Item {i} in pairs must be a (premise, hypothesis) 2-tuple."
                )
            p = self._validate_input_text(pair[0], f"Premise at index {i}")
            h = self._validate_input_text(pair[1], f"Hypothesis at index {i}")
            validated_pairs.append((p, h))

        if not self.is_loaded:
            self.load_model()

        assert self._tokenizer is not None
        assert self._model is not None
        assert self._label_indices is not None

        effective_batch_size = batch_size or self.config.batch_size
        results: List[NLIResult] = []

        import torch

        contra_idx = self._label_indices[NLILabel.CONTRADICTION.value]
        neut_idx = self._label_indices[NLILabel.NEUTRAL.value]
        entail_idx = self._label_indices[NLILabel.ENTAILMENT.value]

        for batch_start in range(0, len(validated_pairs), effective_batch_size):
            batch_pairs = validated_pairs[batch_start : batch_start + effective_batch_size]
            premises = [bp[0] for bp in batch_pairs]
            hypotheses = [bp[1] for bp in batch_pairs]

            t_start = time.perf_counter()

            # Tokenize pair using controlled truncation strategy
            inputs = self._tokenizer(
                premises,
                hypotheses,
                max_length=self.config.max_length,
                truncation=self.config.truncation_strategy,
                padding=True,
                return_tensors="pt",
            )

            if self.config.device != "cpu":
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits

            batch_latency_ms = (time.perf_counter() - t_start) * 1000.0
            per_item_latency_ms = batch_latency_ms / len(batch_pairs)

            # Convert logits to CPU and extract probabilities
            if hasattr(logits, "detach"):
                logits_list = logits.detach().cpu().tolist()
            else:
                logits_list = logits.tolist() if hasattr(logits, "tolist") else list(logits)

            logits_list = logits_list[:len(batch_pairs)]

            for idx, raw_logits in enumerate(logits_list):
                probs = _softmax(raw_logits)

                prob_contra = float(probs[contra_idx])
                prob_neut = float(probs[neut_idx])
                prob_entail = float(probs[entail_idx])

                prob_dict = {
                    NLILabel.CONTRADICTION.value: prob_contra,
                    NLILabel.NEUTRAL.value: prob_neut,
                    NLILabel.ENTAILMENT.value: prob_entail,
                }

                # Predicted label is the class with highest probability
                pred_label = max(prob_dict, key=prob_dict.get)  # type: ignore

                premise_orig, hyp_orig = batch_pairs[idx]
                results.append(
                    NLIResult(
                        premise=premise_orig,
                        hypothesis=hyp_orig,
                        language=canon_lang,
                        predicted_label=pred_label,
                        entailment_prob=prob_entail,
                        neutral_prob=prob_neut,
                        contradiction_prob=prob_contra,
                        probabilities=prob_dict,
                        model_id=self.config.model_id,
                        latency_ms=round(per_item_latency_ms, 2),
                    )
                )

        return results
