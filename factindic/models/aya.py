"""Aya Expanse / Cohere Command A+ inference client supporting hosted providers.

Strict Policy:
- Uses environment variables for credentials (never hardcodes or logs secrets).
- Emits clear diagnostic errors when credentials are not configured.
- Never silently switches models or alters generation parameters.
"""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

from factindic.models.alternative import load_local_factindic_env
from factindic.models.base import BaseLLMClient, GenerationResult

# Primary model identifier: Cohere Command A+ (05-2026 release)
COMMAND_A_PLUS_MODEL_ID = "command-a-plus-05-2026"
PRIMARY_MODEL_ID = COMMAND_A_PLUS_MODEL_ID
COHERE_DEFAULT_MODEL_ID = PRIMARY_MODEL_ID

# Historical / deprecated references preserved for backward compatibility & audit logs
AYA_MODEL_ID_COHERE = "c4ai-aya-expanse-8b"
AYA_MODEL_ID_HF = "CohereForAI/aya-expanse-8b"


class AyaExpanseClient(BaseLLMClient):
    """Inference client for FactIndic generation models (Cohere Command A+ primary)."""

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        model_id: Optional[str] = None,
    ):
        """Initialize the inference client.

        Args:
            provider: 'cohere', 'huggingface', 'openai_compatible', or None for auto-detection.
            api_key: Optional override for the provider API key (prefers env vars).
            base_url: Optional base URL for custom or OpenAI-compatible endpoints.
            timeout: HTTP request timeout in seconds.
            model_id: Optional model identifier override (defaults to PRIMARY_MODEL_ID for Cohere).
        """
        self.timeout = timeout
        self.provider, self.api_key, self.base_url, self.model_id = self._resolve_provider(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model_id=model_id,
        )

    def _resolve_provider(
        self,
        provider: Optional[str],
        api_key: Optional[str],
        base_url: Optional[str],
        model_id: Optional[str] = None,
    ) -> Tuple[str, str, str, str]:
        """Resolve active provider and credentials from environment variables."""
        local_env = load_local_factindic_env()

        # 1. Cohere API (Primary)
        cohere_key = (
            api_key
            or os.environ.get("COHERE_API_KEY")
            or os.environ.get("CO_API_KEY")
            or local_env.get("COHERE_API_KEY")
            or local_env.get("CO_API_KEY")
        )
        if (provider == "cohere" or provider is None) and cohere_key:
            active_model = model_id or COHERE_DEFAULT_MODEL_ID
            return "cohere", cohere_key, "https://api.cohere.com/v2/chat", active_model

        # 2. Hugging Face Inference API
        hf_key = (
            api_key
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_API_KEY")
            or local_env.get("HF_TOKEN")
            or local_env.get("HUGGINGFACE_API_KEY")
        )
        if (provider == "huggingface" or provider is None) and hf_key:
            active_model = model_id or AYA_MODEL_ID_HF
            hf_url = f"https://api-inference.huggingface.co/models/{active_model}"
            return "huggingface", hf_key, hf_url, active_model

        # 3. Custom OpenAI-compatible endpoint (e.g., Together AI, vLLM, OpenRouter)
        custom_base = (
            base_url
            or os.environ.get("AYA_API_BASE")
            or os.environ.get("OPENAI_API_BASE")
            or local_env.get("AYA_API_BASE")
            or local_env.get("OPENAI_API_BASE")
        )
        custom_key = (
            api_key
            or os.environ.get("AYA_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or local_env.get("AYA_API_KEY")
            or local_env.get("OPENAI_API_KEY")
        )
        if (provider == "openai_compatible" or provider is None) and custom_base and custom_key:
            active_model = model_id or PRIMARY_MODEL_ID
            endpoint = custom_base.rstrip("/") + "/chat/completions"
            return "openai_compatible", custom_key, endpoint, active_model

        # No valid provider/credentials detected
        raise RuntimeError(
            "No viable inference credentials found for primary generation model (Cohere Command A+).\n"
            "Supported providers:\n"
            "  1. Cohere API (Primary Recommended): Set environment variable 'COHERE_API_KEY'\n"
            "  2. Hugging Face Inference API: Set environment variable 'HF_TOKEN'\n"
            "  3. Custom Endpoint: Set 'AYA_API_BASE' and 'AYA_API_KEY'\n"
            "Please export one of the above credentials before initiating generation."
        )

    def _call_cohere(
        self,
        prompt: str,
        temperature: float,
        max_new_tokens: int,
        top_p: float,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "p": top_p,
        }
        if response_format:
            payload["response_format"] = response_format
        else:
            payload["thinking"] = {"type": "disabled"}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "FactIndic-Research/1.0",
            },
        )
        max_retries = 5
        backoff = 3.0
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    # Cohere V2 format: content array may contain thinking and text blocks
                    content = res_data.get("message", {}).get("content", [])
                    if content and isinstance(content, list):
                        text_parts = [
                            item["text"]
                            for item in content
                            if isinstance(item, dict) and "text" in item and item.get("type") != "thinking"
                        ]
                        if text_parts:
                            return "\n".join(text_parts).strip()
                        if "text" in content[0]:
                            return content[0]["text"].strip()
                    raise ValueError(f"Unexpected Cohere response structure: {res_data}")
            except urllib.error.HTTPError as http_err:
                body_err = ""
                try:
                    body_err = http_err.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                if "1000 api calls / month" in body_err.lower() or ("trial key" in body_err.lower() and http_err.code == 429):
                    raise http_err
                if http_err.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    retry_after = http_err.headers.get("Retry-After")
                    try:
                        sleep_dur = float(retry_after) if retry_after else (backoff * (2 ** attempt))
                    except (ValueError, TypeError):
                        sleep_dur = backoff * (2 ** attempt)
                    time.sleep(sleep_dur)
                    continue
                raise

    def _call_huggingface(
        self,
        prompt: str,
        temperature: float,
        max_new_tokens: int,
        top_p: float,
    ) -> str:
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "top_p": top_p,
                "return_full_text": False,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "FactIndic-Research/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            if isinstance(res_data, list) and len(res_data) > 0 and "generated_text" in res_data[0]:
                return res_data[0]["generated_text"].strip()
            raise ValueError(f"Unexpected Hugging Face response structure: {res_data}")

    def _call_openai_compatible(
        self,
        prompt: str,
        temperature: float,
        max_new_tokens: int,
        top_p: float,
    ) -> str:
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "top_p": top_p,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "FactIndic-Research/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            choices = res_data.get("choices", [])
            if choices and "message" in choices[0] and "content" in choices[0]["message"]:
                return choices[0]["message"]["content"].strip()
            raise ValueError(f"Unexpected OpenAI-compatible response structure: {res_data}")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_new_tokens: int = 256,
        top_p: float = 0.9,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Execute text generation with the configured hosted provider."""
        t0 = time.time()
        params = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
        }
        if response_format:
            params["response_format"] = response_format

        if self.provider == "cohere":
            text = self._call_cohere(
                prompt,
                temperature,
                max_new_tokens,
                top_p,
                response_format=response_format,
            )
        elif self.provider == "huggingface":
            text = self._call_huggingface(prompt, temperature, max_new_tokens, top_p)
        elif self.provider == "openai_compatible":
            text = self._call_openai_compatible(prompt, temperature, max_new_tokens, top_p)
        else:
            raise NotImplementedError(f"Unsupported provider: {self.provider}")

        latency = round(time.time() - t0, 3)

        return GenerationResult(
            text=text,
            model_id=self.model_id,
            provider=self.provider,
            generation_parameters=params,
            latency_seconds=latency,
        )


# Backward compatibility and ergonomic aliases
CohereCommandClient = AyaExpanseClient
CohereClient = AyaExpanseClient

