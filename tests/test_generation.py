import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from factindic.generation import (
    GenerationRecord,
    load_generation_records,
    run_article_generation,
    save_generation_records,
)
from factindic.models.aya import (
    AyaExpanseClient,
    CohereCommandClient,
    COMMAND_A_PLUS_MODEL_ID,
    PRIMARY_MODEL_ID,
)
from factindic.models.base import BaseLLMClient, GenerationResult
from factindic.prompts.templates import PromptStrategy
from factindic.schema import FactIndicRecord


class MockLLMClient(BaseLLMClient):
    """Mock LLM client returning deterministic results for testing."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.model_id = "test-mock-aya"
        self.provider = "mock"

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_new_tokens: int = 256,
        top_p: float = 0.9,
    ) -> GenerationResult:
        if self.should_fail:
            raise ConnectionError("Mock network connection timeout.")
        return GenerationResult(
            text="This is a generated concise summary.",
            model_id=self.model_id,
            provider=self.provider,
            generation_parameters={
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "top_p": top_p,
            },
            latency_seconds=0.123,
        )


def test_generation_record_serialization_roundtrip(tmp_path: Path):
    target = tmp_path / "generations_test.jsonl"
    record = GenerationRecord(
        source_dataset="csebuetnlp/xlsum",
        source_config="english",
        source_split="test",
        source_record_id="rec_101",
        language="en",
        source_text="This is the full source article text.",
        reference_summary="This is the reference summary.",
        prompt_strategy="prompt_a",
        prompt_template_version="v1.0",
        model_id="c4ai-aya-expanse-8b",
        provider="cohere",
        generation_parameters={"temperature": 0.3, "max_new_tokens": 256, "top_p": 0.9},
        generated_summary="Generated summary output.",
        generation_timestamp_utc="2026-10-22T00:00:00Z",
        latency_seconds=0.45,
        status="SUCCESS",
    )

    written = save_generation_records([record], target)
    assert written == 1

    loaded = load_generation_records(target)
    assert len(loaded) == 1
    assert loaded[0].source_record_id == "rec_101"
    assert loaded[0].prompt_strategy == "prompt_a"
    assert loaded[0].status == "SUCCESS"


def test_run_article_generation_success():
    article = FactIndicRecord(
        example_id="xlsum_en_123",
        language="en",
        source_text="Scientists at Cambridge have demonstrated an experimental battery.",
        reference_summary="Cambridge researchers developed a battery.",
        metadata={"raw_id": "123", "hf_config": "english", "split": "test"},
    )
    client = MockLLMClient(should_fail=False)

    rec = run_article_generation(
        record=article,
        client=client,
        strategy=PromptStrategy.PROMPT_A,
    )

    assert rec.status == "SUCCESS"
    assert rec.generated_summary == "This is a generated concise summary."
    assert rec.source_record_id == "123"
    assert rec.prompt_strategy == "prompt_a"
    assert rec.error_message is None


def test_run_article_generation_failure_handling():
    article = FactIndicRecord(
        example_id="xlsum_hi_456",
        language="hi",
        source_text="भारतीय अंतरिक्ष अनुसंधान संगठन ने नया परीक्षण किया।",
        reference_summary="इसरो ने परीक्षण किया।",
        metadata={"raw_id": "456", "hf_config": "hindi", "split": "test"},
    )
    client = MockLLMClient(should_fail=True)

    rec = run_article_generation(
        record=article,
        client=client,
        strategy=PromptStrategy.PROMPT_B,
    )

    assert rec.status == "FAILED"
    assert rec.generated_summary == ""
    assert "Mock network connection timeout" in rec.error_message
    assert rec.prompt_strategy == "prompt_b"


def test_aya_client_credential_guard(monkeypatch):
    # Ensure no API keys are present in env or local .env
    monkeypatch.setattr("factindic.models.aya.load_local_factindic_env", lambda: {})
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("CO_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    monkeypatch.delenv("AYA_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    with pytest.raises(RuntimeError, match="No viable inference credentials found"):
        AyaExpanseClient()


def test_aya_client_loads_from_local_env(monkeypatch):
    """Verify AyaExpanseClient reads COHERE_API_KEY from local .env when env vars are unset."""
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("CO_API_KEY", raising=False)
    monkeypatch.setattr(
        "factindic.models.aya.load_local_factindic_env",
        lambda: {"COHERE_API_KEY": "env_file_cohere_key"},
    )
    client = AyaExpanseClient()
    assert client.provider == "cohere"
    assert client.api_key == "env_file_cohere_key"
    assert client.model_id == "command-a-plus-05-2026"



def test_cohere_client_model_selection_default():
    """Verify AyaExpanseClient / CohereCommandClient defaults to Cohere Command A+."""
    client = AyaExpanseClient(provider="cohere", api_key="dummy_cohere_key")
    assert client.provider == "cohere"
    assert client.model_id == "command-a-plus-05-2026"
    assert client.base_url == "https://api.cohere.com/v2/chat"
    assert client.api_key == "dummy_cohere_key"

    # Also test via CohereCommandClient alias
    alias_client = CohereCommandClient(provider="cohere", api_key="dummy_cohere_key")
    assert alias_client.model_id == "command-a-plus-05-2026"
    assert alias_client.provider == "cohere"


def test_cohere_client_model_override():
    """Verify model_id can be overridden during client initialization."""
    client = AyaExpanseClient(
        provider="cohere",
        api_key="dummy_cohere_key",
        model_id="custom-cohere-model",
    )
    assert client.model_id == "custom-cohere-model"


def test_cohere_request_payload_and_response_parsing():
    """Verify exact payload structure and response parsing for Cohere V2 chat endpoint."""
    client = AyaExpanseClient(provider="cohere", api_key="test_cohere_api_key_123")

    mock_response_data = {
        "id": "res-uuid-12345",
        "finish_reason": "COMPLETE",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "This is a factually grounded test summary from Command A+.",
                }
            ],
        },
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_response_data).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value = mock_resp
        result = client.generate(
            prompt="Summarize this text accurately.",
            temperature=0.3,
            max_new_tokens=256,
            top_p=0.9,
        )

        # 1. Verify call was made
        assert mock_urlopen.call_count == 1
        req = mock_urlopen.call_args[0][0]

        # 2. Verify URL, headers, and payload
        assert req.full_url == "https://api.cohere.com/v2/chat"
        assert req.headers["Authorization"] == "Bearer test_cohere_api_key_123"
        assert req.headers["Content-type"] == "application/json"

        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "command-a-plus-05-2026"
        assert payload["messages"] == [{"role": "user", "content": "Summarize this text accurately."}]
        assert payload["temperature"] == 0.3
        assert payload["max_tokens"] == 256
        assert payload["p"] == 0.9
        assert payload["thinking"] == {"type": "disabled"}

        # 3. Verify GenerationResult
        assert isinstance(result, GenerationResult)
        assert result.text == "This is a factually grounded test summary from Command A+."
        assert result.model_id == "command-a-plus-05-2026"
        assert result.provider == "cohere"
        assert result.generation_parameters == {
            "temperature": 0.3,
            "max_new_tokens": 256,
            "top_p": 0.9,
        }


def test_dual_prompting_identical_parameters_with_cohere():
    """Verify Prompt A and Prompt B use the same model and identical parameters across languages."""
    client = AyaExpanseClient(provider="cohere", api_key="dummy_key")

    mock_response_data = {
        "message": {
            "content": [{"text": "Summary output."}],
        }
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_response_data).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    # English article
    article_en = FactIndicRecord(
        example_id="xlsum_en_1",
        language="en",
        source_text="English source article text for testing.",
        reference_summary="English reference.",
        metadata={"raw_id": "en_1", "hf_config": "english", "split": "test"},
    )
    # Hindi article
    article_hi = FactIndicRecord(
        example_id="xlsum_hi_1",
        language="hi",
        source_text="परीक्षण के लिए हिंदी स्रोत लेख पाठ।",
        reference_summary="हिंदी संदर्भ।",
        metadata={"raw_id": "hi_1", "hf_config": "hindi", "split": "test"},
    )

    with patch("urllib.request.urlopen", return_value=mock_resp):
        # Prompt A on EN
        rec_a_en = run_article_generation(article_en, client, PromptStrategy.PROMPT_A)
        # Prompt B on EN
        rec_b_en = run_article_generation(article_en, client, PromptStrategy.PROMPT_B)
        # Prompt A on HI
        rec_a_hi = run_article_generation(article_hi, client, PromptStrategy.PROMPT_A)
        # Prompt B on HI
        rec_b_hi = run_article_generation(article_hi, client, PromptStrategy.PROMPT_B)

    # Check identical model ID and parameters
    for rec in [rec_a_en, rec_b_en, rec_a_hi, rec_b_hi]:
        assert rec.status == "SUCCESS"
        assert rec.model_id == "command-a-plus-05-2026"
        assert rec.provider == "cohere"
        assert rec.generation_parameters == {
            "temperature": 0.3,
            "max_new_tokens": 256,
            "top_p": 0.9,
        }

    assert rec_a_en.prompt_strategy == "prompt_a"
    assert rec_b_en.prompt_strategy == "prompt_b"
    assert rec_a_hi.prompt_strategy == "prompt_a"
    assert rec_b_hi.prompt_strategy == "prompt_b"

