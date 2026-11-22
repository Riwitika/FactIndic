"""Multilingual Natural Language Inference (NLI) verification adapter.

Uses joeddav/xlm-roberta-large-xnli to evaluate premise-hypothesis entailment for
factual consistency evaluation across English and Hindi summary sentences.

Note on Factuality Assessment:
    Entailment probability serves as an empirical heuristic proxy for factual
    consistency with source evidence. It should NOT be treated as a definitive
    or ground-truth measure of real-world factual correctness, as models are subject
    to premise-length biases, lack of real-world knowledge, and linguistic domain shifts.
"""

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from factindic.cache import ResponseCache
from factindic.exceptions import (
    AuthenticationError,
    InferenceTimeoutError,
    QuotaExhaustedError,
    RateLimitError,
)
from factindic.models.alternative import load_local_factindic_env
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

    def __init__(
        self,
        config: Optional[NLIConfig] = None,
        cache: Optional[ResponseCache] = None,
    ) -> None:
        """Initialize the XNLIVerifier with configuration.

        Args:
            config: Optional NLIConfig instance. Uses default CPU settings if omitted.
            cache: Optional ResponseCache for persistent caching of verification calls.
        """
        self.config = config or NLIConfig()
        self.cache = cache
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._label_indices: Optional[Dict[str, int]] = None
        self._api_key: Optional[str] = None

    def _resolve_api_key(self) -> Optional[str]:
        """Resolve hosted API key from config or environment variables."""
        if self.config.api_key:
            return self.config.api_key
        local_env = load_local_factindic_env()
        return (
            os.environ.get("COHERE_API_KEY")
            or os.environ.get("CO_API_KEY")
            or local_env.get("COHERE_API_KEY")
            or local_env.get("CO_API_KEY")
        )

    @property
    def is_loaded(self) -> bool:
        """Check whether the verifier is ready for inference."""
        if self.config.provider in ("cohere", "hosted"):
            return self._api_key is not None
        return self._model is not None and self._tokenizer is not None

    def load_model(self) -> None:
        """Explicitly load the tokenizer and model into memory or validate hosted credentials.

        Raises:
            VerificationDependencyError: If torch or transformers is not installed for local provider.
            VerificationModelLoadError: If loading from Hugging Face fails or API credentials are missing.
        """
        if self.config.provider in ("cohere", "hosted"):
            self._api_key = self._resolve_api_key()
            if not self._api_key:
                raise VerificationModelLoadError(
                    "Hosted NLI verification requires an API key. Set COHERE_API_KEY environment variable."
                )
            return

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
        self._api_key = None
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

        if self.config.provider in ("cohere", "hosted"):
            return self._verify_batch_hosted(validated_pairs, canon_lang)

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
                nli_res = NLIResult(
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
                if self.cache is not None:
                    self.cache.set_nli(premise_orig, hyp_orig, canon_lang, self.config.model_id, nli_res.to_dict())
                results.append(nli_res)

        return results

    def _verify_batch_hosted(
        self,
        pairs: List[Tuple[str, str]],
        language: str,
    ) -> List[NLIResult]:
        """Evaluate premise-hypothesis pairs using hosted multilingual NLI."""
        if not self._api_key:
            self._api_key = self._resolve_api_key()
        if not self._api_key:
            raise VerificationModelLoadError(
                "Hosted NLI inference requires an API key. Set COHERE_API_KEY environment variable."
            )

        results: List[NLIResult] = []
        schema = {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": ["entailment", "neutral", "contradiction"]},
                "entailment_prob": {"type": "number"},
                "neutral_prob": {"type": "number"},
                "contradiction_prob": {"type": "number"},
            },
            "required": ["label", "entailment_prob", "neutral_prob", "contradiction_prob"],
        }
        url = self.config.api_base_url or "https://api.cohere.com/v2/chat"

        for p, h in pairs:
            # Check cache first
            if self.cache is not None:
                cached_dict = self.cache.get_nli(p, h, language, self.config.model_id)
                if cached_dict is not None:
                    results.append(NLIResult.from_dict(cached_dict))
                    continue

            prompt = (
                f"You are a rigorous Natural Language Inference (NLI) evaluator.\n"
                f"Given a premise and a hypothesis in {language}, determine whether the hypothesis is:\n"
                f"- entailment: The premise strictly supports or entails the hypothesis.\n"
                f"- contradiction: The hypothesis contradicts or is incompatible with the premise.\n"
                f"- neutral: The premise neither entails nor contradicts the hypothesis (unsupported, plausible, or missing info).\n\n"
                f"Language: {language}\n"
                f"Premise: {p}\n"
                f"Hypothesis: {h}\n"
            )
            payload = {
                "model": self.config.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 4096,
                "response_format": {"type": "json_object", "schema": schema},
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "FactIndic-Research/1.0",
                },
            )

            t0 = time.perf_counter()
            last_exc: Optional[Exception] = None
            out_data: Optional[Dict[str, Any]] = None

            for attempt in range(self.config.max_retries):
                try:
                    with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                        res_json = json.loads(resp.read().decode("utf-8"))
                        text_val = ""
                        for item in res_json.get("message", {}).get("content", []):
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_val = item.get("text", "")
                                break
                        if not text_val and "text" in res_json.get("message", {}).get("content", [{}])[0]:
                            text_val = res_json["message"]["content"][0]["text"]

                        cleaned = text_val.strip()
                        if cleaned.startswith("```"):
                            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                            cleaned = re.sub(r"\s*```$", "", cleaned)
                        out_data = json.loads(cleaned)
                        break
                except Exception as exc:
                    last_exc = exc
                    err_msg = str(exc).lower()
                    status_code = getattr(exc, "code", None)
                    resp_body = ""
                    if isinstance(exc, urllib.error.HTTPError):
                        try:
                            resp_body = exc.read().decode("utf-8", errors="replace")
                            err_msg += " " + resp_body.lower()
                        except Exception:
                            pass

                    # 1. Monthly quota exhausted: fail fast immediately
                    if (
                        "1000 api calls / month" in err_msg
                        or ("trial key" in err_msg and "429" in err_msg)
                        or "monthly quota" in err_msg
                    ):
                        raise QuotaExhaustedError(
                            f"Cohere trial key monthly quota of 1000 API calls exceeded: {exc}",
                            response_body=resp_body,
                        ) from exc

                    # 2. Authentication error: fail fast immediately
                    if status_code in (401, 403) or "unauthorized" in err_msg or "invalid api key" in err_msg:
                        raise AuthenticationError(
                            f"Hosted NLI authentication failed: {exc}",
                            status_code=status_code,
                            response_body=resp_body,
                        ) from exc

                    # 3. Timeout error
                    if isinstance(exc, (TimeoutError, urllib.error.URLError)) and "timed out" in err_msg:
                        if attempt == self.config.max_retries - 1:
                            raise InferenceTimeoutError(
                                f"Hosted NLI inference timed out after {self.config.timeout}s: {exc}"
                            ) from exc

                    # 4. Burst rate limit (429) - retry with backoff or Retry-After
                    wait_time = 2.0 * (attempt + 1)
                    if status_code == 429:
                        if attempt == self.config.max_retries - 1:
                            raise RateLimitError(
                                f"Hosted NLI rate limit exceeded after {self.config.max_retries} retries: {exc}",
                                status_code=429,
                                response_body=resp_body,
                            ) from exc
                        retry_after = exc.headers.get("Retry-After") if hasattr(exc, "headers") else None
                        wait_time = float(retry_after) if retry_after else max(15.0, 10.0 * (attempt + 1))
                    elif "429" in str(exc) or "too many requests" in str(exc).lower():
                        wait_time = max(15.0, 10.0 * (attempt + 1))

                    time.sleep(wait_time)

            if out_data is None:
                raise RuntimeError(
                    f"Hosted NLI inference failed after {self.config.max_retries} attempts: {last_exc}"
                )

            latency_ms = (time.perf_counter() - t0) * 1000.0

            raw_label = str(out_data.get("label", "neutral")).lower().strip()
            entail_prob = float(out_data.get("entailment_prob", 0.0))
            neut_prob = float(out_data.get("neutral_prob", 0.0))
            contra_prob = float(out_data.get("contradiction_prob", 0.0))

            # Normalize probabilities if sum > 0
            total_prob = entail_prob + neut_prob + contra_prob
            if total_prob > 0:
                entail_prob /= total_prob
                neut_prob /= total_prob
                contra_prob /= total_prob
            else:
                entail_prob, neut_prob, contra_prob = 0.333, 0.334, 0.333

            prob_dict = {
                NLILabel.CONTRADICTION.value: contra_prob,
                NLILabel.NEUTRAL.value: neut_prob,
                NLILabel.ENTAILMENT.value: entail_prob,
            }

            if raw_label not in [l.value for l in NLILabel]:
                raw_label = max(prob_dict, key=prob_dict.get)  # type: ignore

            nli_res = NLIResult(
                premise=p,
                hypothesis=h,
                language=language,
                predicted_label=raw_label,
                entailment_prob=round(entail_prob, 4),
                neutral_prob=round(neut_prob, 4),
                contradiction_prob=round(contra_prob, 4),
                probabilities={k: round(v, 4) for k, v in prob_dict.items()},
                model_id=self.config.model_id,
                latency_ms=round(latency_ms, 2),
            )

            # Store in cache
            if self.cache is not None:
                self.cache.set_nli(p, h, language, self.config.model_id, nli_res.to_dict())

            results.append(nli_res)

            if len(pairs) > 1:
                time.sleep(0.3)

        return results
