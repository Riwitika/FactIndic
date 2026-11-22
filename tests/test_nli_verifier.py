"""Unit tests for the FactIndic multilingual NLI verification adapter.

All tests run 100% offline using mocked model and tokenizer outputs without
downloading model weights or requiring network access.
"""

import json
import math
import sys
from types import ModuleType
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from factindic.verification import (
    NLIConfig,
    NLILabel,
    NLIResult,
    SUPPORTED_VERIFICATION_LANGUAGES,
    VerificationDependencyError,
    VerificationModelLoadError,
    XNLIVerifier,
)
from factindic.verification.nli import _softmax


class DummyConfig:
    """Mock Hugging Face model configuration."""

    def __init__(self, id2label: Any = "DEFAULT", label2id: Any = "DEFAULT"):
        if id2label == "DEFAULT":
            self.id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
        else:
            self.id2label = id2label

        if label2id == "DEFAULT":
            self.label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
        else:
            self.label2id = label2id


class DummyOutputs:
    """Mock Hugging Face model forward pass outputs."""

    def __init__(self, logits: Any):
        self.logits = logits


class DummyModel:
    """Mock PyTorch sequence classification model."""

    def __init__(self, config: DummyConfig = None, logits: Any = None):
        self.config = config or DummyConfig()
        self._initial_logits = list(logits) if logits is not None else [[-1.0, 0.0, 4.0]]
        self.logits = list(self._initial_logits)

    def to(self, device: str) -> "DummyModel":
        return self

    def eval(self) -> "DummyModel":
        return self

    def __call__(self, **kwargs: Any) -> DummyOutputs:
        input_ids = kwargs.get("input_ids", [])
        n = len(input_ids) if input_ids else len(self.logits)
        if len(self._initial_logits) == 1:
            return DummyOutputs(self._initial_logits * n)
        if len(self.logits) >= n:
            ret = self.logits[:n]
            self.logits = self.logits[n:]
            return DummyOutputs(ret)
        return DummyOutputs(self._initial_logits[:n])


class DummyTokenizer:
    """Mock Hugging Face tokenizer."""

    def __init__(self):
        self.last_kwargs = {}

    def __call__(
        self,
        premises: List[str],
        hypotheses: List[str],
        max_length: int = 512,
        truncation: Any = "only_first",
        padding: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, Any]:
        self.last_kwargs = {
            "premises": premises,
            "hypotheses": hypotheses,
            "max_length": max_length,
            "truncation": truncation,
            "padding": padding,
            "return_tensors": return_tensors,
        }
        return {
            "input_ids": [[101, 102]] * len(premises),
            "attention_mask": [[1, 1]] * len(premises),
        }


def _ensure_mock_torch_and_transformers():
    """Ensure torch and transformers exist in sys.modules with minimal mock interfaces."""
    if "torch" not in sys.modules:
        mock_torch = ModuleType("torch")
        mock_torch.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
        mock_torch.cuda = MagicMock(is_available=MagicMock(return_value=False))
        sys.modules["torch"] = mock_torch

    if "transformers" not in sys.modules:
        mock_transformers = ModuleType("transformers")
        mock_transformers.AutoTokenizer = MagicMock()
        mock_transformers.AutoModelForSequenceClassification = MagicMock()
        sys.modules["transformers"] = mock_transformers


@pytest.fixture(autouse=True)
def setup_mock_modules():
    _ensure_mock_torch_and_transformers()


# ---------------------------------------------------------------------------
# Test 1: Lazy loading & unload
# ---------------------------------------------------------------------------

def test_lazy_loading():
    """Verifier must not load model or tokenizer during initialization."""
    verifier = XNLIVerifier()
    assert not verifier.is_loaded
    assert verifier._model is None
    assert verifier._tokenizer is None


def test_unload_model():
    """unload_model must reset internal state and mark verifier as not loaded."""
    verifier = XNLIVerifier()
    verifier._model = DummyModel()
    verifier._tokenizer = DummyTokenizer()
    verifier._label_indices = {"entailment": 2, "neutral": 1, "contradiction": 0}
    assert verifier.is_loaded

    verifier.unload_model()
    assert not verifier.is_loaded
    assert verifier._model is None
    assert verifier._tokenizer is None
    assert verifier._label_indices is None


# ---------------------------------------------------------------------------
# Test 2: Label mapping (standard & permuted)
# ---------------------------------------------------------------------------

def test_correct_label_mapping_standard():
    """Verifier dynamically maps standard joeddav/xlm-roberta-large-xnli label indices."""
    verifier = XNLIVerifier()
    config = DummyConfig(id2label={0: "contradiction", 1: "neutral", 2: "entailment"})
    mapping = verifier._resolve_label_indices(config)

    assert mapping[NLILabel.CONTRADICTION.value] == 0
    assert mapping[NLILabel.NEUTRAL.value] == 1
    assert mapping[NLILabel.ENTAILMENT.value] == 2


def test_correct_label_mapping_permuted():
    """Verifier correctly resolves arbitrary or permuted id2label configurations."""
    verifier = XNLIVerifier()
    config = DummyConfig(id2label={0: "entailment", 1: "contradiction", 2: "neutral"})
    mapping = verifier._resolve_label_indices(config)

    assert mapping[NLILabel.ENTAILMENT.value] == 0
    assert mapping[NLILabel.CONTRADICTION.value] == 1
    assert mapping[NLILabel.NEUTRAL.value] == 2


def test_correct_label_mapping_from_label2id():
    """Verifier resolves mapping when model provides label2id instead of id2label."""
    verifier = XNLIVerifier()
    config = DummyConfig(id2label=None, label2id={"LABEL_ENTAILMENT": 1, "LABEL_CONTRADICTION": 2, "NEUTRAL": 0})
    mapping = verifier._resolve_label_indices(config)

    assert mapping[NLILabel.NEUTRAL.value] == 0
    assert mapping[NLILabel.ENTAILMENT.value] == 1
    assert mapping[NLILabel.CONTRADICTION.value] == 2


# ---------------------------------------------------------------------------
# Test 3: Probability normalization
# ---------------------------------------------------------------------------

def test_probability_normalization():
    """Probabilities over the 3 NLI classes must strictly sum to 1.0."""
    logits = [-2.4, 0.8, 3.1]
    probs = _softmax(logits)
    assert len(probs) == 3
    assert sum(probs) == pytest.approx(1.0, rel=1e-5)
    for p in probs:
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Test 4: Predictions (Entailment, Contradiction, Neutral)
# ---------------------------------------------------------------------------

def test_entailment_prediction():
    """High entailment logit produces entailment prediction with highest probability."""
    verifier = XNLIVerifier()
    # 0: contradiction (-3.0), 1: neutral (-1.0), 2: entailment (5.0)
    verifier._model = DummyModel(logits=[[-3.0, -1.0, 5.0]])
    verifier._tokenizer = DummyTokenizer()
    verifier._label_indices = {"contradiction": 0, "neutral": 1, "entailment": 2}

    result = verifier.verify_premise_hypothesis(
        premise="The BBC reported record monsoon rainfall in Mumbai on Tuesday.",
        hypothesis="Mumbai experienced heavy monsoon rain.",
        language="en",
    )

    assert result.predicted_label == NLILabel.ENTAILMENT.value
    assert result.entailment_prob > 0.95
    assert result.entailment_prob > result.neutral_prob
    assert result.entailment_prob > result.contradiction_prob
    assert sum(result.probabilities.values()) == pytest.approx(1.0, rel=1e-5)


def test_contradiction_prediction():
    """High contradiction logit produces contradiction prediction with highest probability."""
    verifier = XNLIVerifier()
    # 0: contradiction (6.0), 1: neutral (0.0), 2: entailment (-2.0)
    verifier._model = DummyModel(logits=[[6.0, 0.0, -2.0]])
    verifier._tokenizer = DummyTokenizer()
    verifier._label_indices = {"contradiction": 0, "neutral": 1, "entailment": 2}

    result = verifier.verify_premise_hypothesis(
        premise="India won the cricket match by 5 wickets in London.",
        hypothesis="India lost the cricket match.",
        language="en",
    )

    assert result.predicted_label == NLILabel.CONTRADICTION.value
    assert result.contradiction_prob > 0.95
    assert result.contradiction_prob > result.entailment_prob


def test_neutral_prediction():
    """High neutral logit produces neutral prediction with highest probability."""
    verifier = XNLIVerifier()
    # 0: contradiction (-1.0), 1: neutral (4.5), 2: entailment (-0.5)
    verifier._model = DummyModel(logits=[[-1.0, 4.5, -0.5]])
    verifier._tokenizer = DummyTokenizer()
    verifier._label_indices = {"contradiction": 0, "neutral": 1, "entailment": 2}

    result = verifier.verify_premise_hypothesis(
        premise="The government announced plans to construct a new highway linking Delhi and Jaipur.",
        hypothesis="The highway will be completed ahead of schedule next year.",
        language="en",
    )

    assert result.predicted_label == NLILabel.NEUTRAL.value
    assert result.neutral_prob > 0.90
    assert result.neutral_prob > result.entailment_prob


# ---------------------------------------------------------------------------
# Test 5: English and Hindi Language Processing
# ---------------------------------------------------------------------------

def test_english_input():
    """Verifier accepts English input using 'en' and 'english' aliases."""
    verifier = XNLIVerifier()
    verifier._model = DummyModel(logits=[[-1.0, 0.0, 3.0]])
    verifier._tokenizer = DummyTokenizer()
    verifier._label_indices = {"contradiction": 0, "neutral": 1, "entailment": 2}

    res1 = verifier.verify_premise_hypothesis("Premise", "Hypothesis", language="en")
    assert res1.language == "en"

    res2 = verifier.verify_premise_hypothesis("Premise", "Hypothesis", language="english")
    assert res2.language == "en"


def test_hindi_input():
    """Verifier accepts Hindi Devanagari input using 'hi' and 'hindi' aliases."""
    verifier = XNLIVerifier()
    verifier._model = DummyModel(logits=[[-2.0, 0.0, 4.0]])
    verifier._tokenizer = DummyTokenizer()
    verifier._label_indices = {"contradiction": 0, "neutral": 1, "entailment": 2}

    res1 = verifier.verify_premise_hypothesis(
        premise="भारतीय वैज्ञानिकों ने अंतरिक्ष में नया उपग्रह सफलतापूर्वक स्थापित किया।",
        hypothesis="उपग्रह का सफल प्रक्षेपण हुआ।",
        language="hi",
    )
    assert res1.language == "hi"
    assert res1.predicted_label == "entailment"

    res2 = verifier.verify_premise_hypothesis(
        premise="भारतीय वैज्ञानिकों ने अंतरिक्ष में नया उपग्रह सफलतापूर्वक स्थापित किया।",
        hypothesis="उपग्रह का सफल प्रक्षेपण हुआ।",
        language="hindi",
    )
    assert res2.language == "hi"


def test_invalid_language():
    """Unsupported languages must raise a clear ValueError."""
    verifier = XNLIVerifier()
    with pytest.raises(ValueError, match="Unsupported verification language 'fr'"):
        verifier.verify_premise_hypothesis("Premise", "Hypothesis", language="fr")


# ---------------------------------------------------------------------------
# Test 6: Empty input validation
# ---------------------------------------------------------------------------

def test_empty_premise():
    """Empty or whitespace-only premise must raise ValueError."""
    verifier = XNLIVerifier()
    with pytest.raises(ValueError, match="Premise at index 0 must be a non-empty string"):
        verifier.verify_premise_hypothesis("", "Valid hypothesis", language="en")

    with pytest.raises(ValueError, match="Premise at index 0 must be a non-empty string"):
        verifier.verify_premise_hypothesis("   \n\t  ", "Valid hypothesis", language="en")


def test_empty_hypothesis():
    """Empty or whitespace-only hypothesis must raise ValueError."""
    verifier = XNLIVerifier()
    with pytest.raises(ValueError, match="Hypothesis at index 0 must be a non-empty string"):
        verifier.verify_premise_hypothesis("Valid premise", "", language="en")

    with pytest.raises(ValueError, match="Hypothesis at index 0 must be a non-empty string"):
        verifier.verify_premise_hypothesis("Valid premise", "   ", language="en")


# ---------------------------------------------------------------------------
# Test 7: Long input handling & truncation
# ---------------------------------------------------------------------------

def test_long_input_handling():
    """Inputs exceeding max_length must invoke tokenizer with controlled truncation."""
    tok = DummyTokenizer()
    verifier = XNLIVerifier(config=NLIConfig(max_length=256, truncation_strategy="only_first"))
    verifier._model = DummyModel(logits=[[0.0, 1.0, 2.0]])
    verifier._tokenizer = tok
    verifier._label_indices = {"contradiction": 0, "neutral": 1, "entailment": 2}

    very_long_premise = "This is a very long premise sentence. " * 200
    short_hypothesis = "Short summary claim."

    result = verifier.verify_premise_hypothesis(
        premise=very_long_premise,
        hypothesis=short_hypothesis,
        language="en",
    )

    assert result is not None
    assert tok.last_kwargs["max_length"] == 256
    assert tok.last_kwargs["truncation"] == "only_first"
    assert tok.last_kwargs["padding"] is True


# ---------------------------------------------------------------------------
# Test 8: Batch inference
# ---------------------------------------------------------------------------

def test_batch_inference():
    """Verifier processes batches across multiple items correctly."""
    verifier = XNLIVerifier(config=NLIConfig(batch_size=2))
    # 3 pairs total: batch 1 has 2 items, batch 2 has 1 item
    verifier._model = DummyModel(logits=[
        [-2.0, 0.0, 3.0],   # pair 0 -> entailment
        [4.0, 0.0, -1.0],   # pair 1 -> contradiction
        [0.0, 5.0, 1.0],    # pair 2 -> neutral
    ])
    verifier._tokenizer = DummyTokenizer()
    verifier._label_indices = {"contradiction": 0, "neutral": 1, "entailment": 2}

    pairs = [
        ("Premise 1", "Hypothesis 1"),
        ("Premise 2", "Hypothesis 2"),
        ("Premise 3", "Hypothesis 3"),
    ]

    results = verifier.verify_batch(pairs, language="en")

    assert len(results) == 3
    assert results[0].predicted_label == "entailment"
    assert results[1].predicted_label == "contradiction"
    assert results[2].predicted_label == "neutral"
    assert all(r.model_id == verifier.config.model_id for r in results)
    assert all(r.latency_ms >= 0.0 for r in results)


def test_batch_inference_empty_list():
    """Empty list of pairs returns empty result without invoking model."""
    verifier = XNLIVerifier()
    assert verifier.verify_batch([], language="en") == []


# ---------------------------------------------------------------------------
# Test 9: Model load failures & missing dependencies
# ---------------------------------------------------------------------------

def test_model_load_failure():
    """VerificationModelLoadError is raised when model from_pretrained fails."""
    verifier = XNLIVerifier()

    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=RuntimeError("Connection timeout")):
        with pytest.raises(VerificationModelLoadError, match="Failed to load NLI model"):
            verifier.load_model()

    assert not verifier.is_loaded


def test_missing_dependencies():
    """VerificationDependencyError is raised when transformers or torch cannot be imported."""
    verifier = XNLIVerifier()

    # Simulate missing transformers by raising ImportError inside load_model
    with patch.dict(sys.modules, {"transformers": None}):
        with pytest.raises(VerificationDependencyError, match="NLI verification requires 'transformers' and 'torch'"):
            verifier.load_model()


# ---------------------------------------------------------------------------
# Test 10: Serialization roundtrip
# ---------------------------------------------------------------------------

def test_nli_result_serialization_roundtrip():
    """NLIResult to_dict and from_dict preserve all attributes and probabilities."""
    orig = NLIResult(
        premise="Source article text",
        hypothesis="Summary claim",
        language="en",
        predicted_label="entailment",
        entailment_prob=0.885,
        neutral_prob=0.100,
        contradiction_prob=0.015,
        probabilities={"contradiction": 0.015, "neutral": 0.100, "entailment": 0.885},
        model_id="joeddav/xlm-roberta-large-xnli",
        latency_ms=45.2,
    )

    d = orig.to_dict()
    restored = NLIResult.from_dict(d)

    assert restored.premise == orig.premise
    assert restored.hypothesis == orig.hypothesis
    assert restored.language == orig.language
    assert restored.predicted_label == orig.predicted_label
    assert restored.entailment_prob == orig.entailment_prob
    assert restored.neutral_prob == orig.neutral_prob
    assert restored.contradiction_prob == orig.contradiction_prob
    assert restored.model_id == orig.model_id
    assert restored.latency_ms == orig.latency_ms
    assert restored.probabilities == orig.probabilities


# ---------------------------------------------------------------------------
# Test 11: Hosted NLI Verification (Cohere provider)
# ---------------------------------------------------------------------------

def test_hosted_nli_loads_with_api_key():
    """Hosted verifier marks is_loaded True when api_key is present."""
    cfg = NLIConfig(provider="cohere", model_id="command-a-plus-05-2026", api_key="test-key-12345")
    verifier = XNLIVerifier(config=cfg)
    assert not verifier.is_loaded
    verifier.load_model()
    assert verifier.is_loaded


def test_hosted_nli_fails_without_api_key(monkeypatch):
    """VerificationModelLoadError is raised when no API key is provided or in environment."""
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("CO_API_KEY", raising=False)
    monkeypatch.delenv("FACTINDIC_COHERE_API_KEY", raising=False)
    with patch("factindic.verification.nli.load_local_factindic_env", return_value={}):
        cfg = NLIConfig(provider="cohere", model_id="command-a-plus-05-2026", api_key=None)
        verifier = XNLIVerifier(config=cfg)
        with pytest.raises(VerificationModelLoadError, match="Hosted NLI verification requires an API key"):
            verifier.load_model()


def test_hosted_nli_verify_batch_mocked():
    """Hosted verify_batch successfully parses JSON responses and normalizes probabilities."""
    cfg = NLIConfig(provider="cohere", model_id="command-a-plus-05-2026", api_key="dummy-key")
    verifier = XNLIVerifier(config=cfg)

    def mock_urlopen(req, timeout=None):
        payload = json.loads(req.data.decode("utf-8"))
        prompt = payload["messages"][0]["content"]
        if "sky is blue" in prompt:
            body = {
                "label": "entailment",
                "entailment_prob": 0.8,
                "neutral_prob": 0.1,
                "contradiction_prob": 0.1,
            }
        else:
            body = {
                "label": "contradiction",
                "entailment_prob": 0.05,
                "neutral_prob": 0.05,
                "contradiction_prob": 0.9,
            }
        resp = MagicMock()
        resp.read.return_value = json.dumps(
            {"message": {"content": [{"type": "text", "text": json.dumps(body)}]}}
        ).encode("utf-8")
        resp.__enter__.return_value = resp
        return resp

    pairs = [
        ("The sky is blue today.", "The sky is blue."),
        ("The cat is sleeping on the mat.", "The cat is barking loudly."),
    ]

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        results = verifier.verify_batch(pairs, language="en")

    assert len(results) == 2
    assert results[0].predicted_label == "entailment"
    assert results[0].entailment_prob == pytest.approx(0.8)
    assert results[0].contradiction_prob == pytest.approx(0.1)
    assert results[0].neutral_prob == pytest.approx(0.1)
    assert results[0].latency_ms > 0

    assert results[1].predicted_label == "contradiction"
    assert results[1].contradiction_prob == pytest.approx(0.9)
    assert results[1].entailment_prob == pytest.approx(0.05)


def test_hosted_nli_verify_batch_normalizes_unnormalized_probs():
    """Hosted verifier normalizes probability dictionaries if they do not sum to 1.0."""
    cfg = NLIConfig(provider="cohere", model_id="command-a-plus-05-2026", api_key="dummy-key")
    verifier = XNLIVerifier(config=cfg)

    mock_resp_json = {
        "label": "neutral",
        "entailment_prob": 10.0,
        "neutral_prob": 80.0,
        "contradiction_prob": 10.0,
    }

    mock_http_response = MagicMock()
    mock_http_response.read.return_value = json.dumps(
        {
            "message": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(mock_resp_json),
                    }
                ]
            }
        }
    ).encode("utf-8")
    mock_http_response.__enter__.return_value = mock_http_response

    with patch("urllib.request.urlopen", return_value=mock_http_response):
        results = verifier.verify_batch([("Premise", "Hypothesis")], language="hi")

    assert len(results) == 1
    assert results[0].predicted_label == "neutral"
    assert results[0].neutral_prob == pytest.approx(0.8)
    assert results[0].entailment_prob == pytest.approx(0.1)
    assert results[0].contradiction_prob == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Test 12: mDeBERTa local verification, caching and revision tracking
# ---------------------------------------------------------------------------

def test_mdeberta_default_config():
    """NLIConfig defaults to MoritzLaurer/mDeBERTa-v3 with pinned revision."""
    cfg = NLIConfig()
    assert cfg.model_id == "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    assert cfg.model_revision == "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"
    assert cfg.device == "cpu"
    assert cfg.provider == "local"
    assert cfg.max_length == 512
    assert cfg.truncation_strategy == "only_first"


def test_mdeberta_standard_id2label_mapping():
    """Verifier maps mDeBERTa native id2label {0: entailment, 1: neutral, 2: contradiction}."""
    verifier = XNLIVerifier()
    config = DummyConfig(id2label={0: "entailment", 1: "neutral", 2: "contradiction"})
    mapping = verifier._resolve_label_indices(config)

    assert mapping[NLILabel.ENTAILMENT.value] == 0
    assert mapping[NLILabel.NEUTRAL.value] == 1
    assert mapping[NLILabel.CONTRADICTION.value] == 2


def test_local_verify_batch_cache_hit_and_miss(tmp_path):
    """Local verify_batch consults SQLite cache, avoiding inference on cache hits."""
    from factindic.cache import ResponseCache

    cache_file = tmp_path / "test_nli_cache.db"
    cache = ResponseCache(db_path=cache_file)
    verifier = XNLIVerifier(cache=cache)

    dummy_model = DummyModel(logits=[[3.0, -1.0, -2.0]])  # Entailment
    dummy_tok = DummyTokenizer()
    verifier._model = dummy_model
    verifier._tokenizer = dummy_tok
    verifier._label_indices = {"entailment": 0, "neutral": 1, "contradiction": 2}

    pair = ("The sky is clear today.", "The sky is blue.")
    results1 = verifier.verify_batch([pair], language="en")
    assert len(results1) == 1
    assert results1[0].predicted_label == "entailment"
    assert cache.get_stats().hits == 0
    assert cache.get_stats().writes == 1

    # Second call should be a cache hit
    results2 = verifier.verify_batch([pair], language="en")
    assert len(results2) == 1
    assert results2[0].predicted_label == "entailment"
    assert cache.get_stats().hits == 1


def test_model_load_passes_revision():
    """load_model passes configured model_revision to Hugging Face from_pretrained."""
    cfg = NLIConfig(
        model_id="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        model_revision="b5113eb38ab63efdd7f280f8c144ea8b13f978ce",
    )
    verifier = XNLIVerifier(config=cfg)

    mock_tok = MagicMock()
    mock_model = MagicMock()
    mock_model.config = DummyConfig(id2label={0: "entailment", 1: "neutral", 2: "contradiction"})

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok) as mock_tok_fn:
        with patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=mock_model) as mock_mod_fn:
            verifier.load_model()
            mock_tok_fn.assert_called_once_with(
                "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                revision="b5113eb38ab63efdd7f280f8c144ea8b13f978ce",
            )
            mock_mod_fn.assert_called_once_with(
                "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                revision="b5113eb38ab63efdd7f280f8c144ea8b13f978ce",
            )

