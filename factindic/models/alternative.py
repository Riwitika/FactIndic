"""Alternative hosted LLM client for temporary demonstration experiments (Phase 2.1).

Strict Policy:
- Work exclusively with credentials present in the local process environment or FactIndic/.env.
- Label all runs explicitly as demo_only (the final research model remains UNDECIDED).
- Never hardcode, print, log, or persist secret API keys.
- Preserve exact model identifier, provider, generation parameters, and UTC timestamps.
- If no credentials exist, halt gracefully without fabricating responses.
"""

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from factindic.models.base import BaseLLMClient, GenerationResult

FACTINDIC_ROOT = Path(__file__).resolve().parent.parent.parent


def load_local_factindic_env() -> Dict[str, str]:
    """Load environment variables strictly from repository root .env if present.

    Does not inspect any files outside the FactIndic repository.
    """
    env_vars: Dict[str, str] = {}
    local_env_file = FACTINDIC_ROOT / ".env"
    if local_env_file.exists() and local_env_file.is_file():
        try:
            for line in local_env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("\"'")
                    if val and not val.startswith("your_"):
                        env_vars[key] = val
        except Exception:
            pass
    return env_vars


class AlternativeDemoClient(BaseLLMClient):
    """Client for accessible hosted models for temporary demo-only experiments."""

    SUPPORTED_PROVIDERS = {
        "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "groq": ["GROQ_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
        "cohere": ["COHERE_API_KEY", "CO_API_KEY"],
        "huggingface": ["HF_TOKEN", "HUGGINGFACE_API_KEY"],
    }

    DEFAULT_MODELS = {
        "google": "gemini-2.5-flash",
        "groq": "llama-3.3-70b-versatile",
        "openai": "gpt-4o-mini",
        "cohere": "command-a-plus-05-2026",
        "huggingface": "CohereForAI/aya-expanse-8b",
    }

    def __init__(
        self,
        provider: Optional[str] = None,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
    ):
        self.timeout = timeout
        local_env = load_local_factindic_env()

        resolved = self._resolve_provider_and_key(
            requested_provider=provider,
            requested_model=model_id,
            explicit_key=api_key,
            local_env=local_env,
        )
        self.provider, self.model_id, self.api_key, self.endpoint = resolved

    def _get_key_from_env(self, var_names: list, local_env: dict) -> Optional[str]:
        for var in var_names:
            val = os.environ.get(var) or local_env.get(var)
            if val and val.strip() and not val.strip().startswith("your_"):
                return val.strip()
        return None

    def _resolve_provider_and_key(
        self,
        requested_provider: Optional[str],
        requested_model: Optional[str],
        explicit_key: Optional[str],
        local_env: dict,
    ) -> Tuple[str, str, str, str]:
        """Resolve an accessible provider from available credentials."""
        # Check providers in priority order
        providers_to_check = [requested_provider] if requested_provider else ["google", "groq", "openai", "cohere", "huggingface"]

        for prov in providers_to_check:
            if prov not in self.SUPPORTED_PROVIDERS:
                continue
            key = explicit_key or self._get_key_from_env(self.SUPPORTED_PROVIDERS[prov], local_env)
            if key:
                model = requested_model or self.DEFAULT_MODELS[prov]
                endpoint = self._get_endpoint(prov, model, key)
                return prov, model, key, endpoint

        # If we reach here, no credentials were found
        raise RuntimeError(
            "No valid API credentials found in environment or FactIndic/.env.\n"
            "To unblock the demo, provide credentials for one of the following providers:\n"
            "  1. Google Gemini (Recommended): export GEMINI_API_KEY='...'\n"
            "  2. Groq Cloud: export GROQ_API_KEY='...'\n"
            "  3. OpenAI: export OPENAI_API_KEY='...'\n"
            "  4. Cohere: export COHERE_API_KEY='...'\n"
            "  5. Hugging Face: export HF_TOKEN='...'\n"
            f"Or save the key into a local .env file inside {FACTINDIC_ROOT / '.env'}"
        )

    def _get_endpoint(self, provider: str, model_id: str, key: str) -> str:
        if provider == "google":
            clean_model = model_id.removeprefix("models/")
            return f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={key}"
        elif provider == "groq":
            return "https://api.groq.com/openai/v1/chat/completions"
        elif provider == "openai":
            return "https://api.openai.com/v1/chat/completions"
        elif provider == "cohere":
            return "https://api.cohere.com/v2/chat"
        elif provider == "huggingface":
            return f"https://api-inference.huggingface.co/models/{model_id}"
        raise ValueError(f"Unknown provider: {provider}")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_new_tokens: int = 256,
        top_p: float = 0.9,
    ) -> GenerationResult:
        """Generate summary completion using the resolved provider."""
        t0 = time.time()
        params = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
        }

        if self.provider == "google":
            text = self._call_google(prompt, temperature, max_new_tokens, top_p)
        elif self.provider in ("groq", "openai"):
            text = self._call_openai_compatible(prompt, temperature, max_new_tokens, top_p)
        elif self.provider == "cohere":
            text = self._call_cohere(prompt, temperature, max_new_tokens, top_p)
        elif self.provider == "huggingface":
            text = self._call_huggingface(prompt, temperature, max_new_tokens, top_p)
        else:
            raise NotImplementedError(f"Provider '{self.provider}' is not supported.")

        latency = round(time.time() - t0, 3)

        return GenerationResult(
            text=text,
            model_id=self.model_id,
            provider=self.provider,
            generation_parameters=params,
            latency_seconds=latency,
        )

    def _call_google(self, prompt: str, temperature: float, max_tokens: int, top_p: float) -> str:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": top_p,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "FactIndic-Demo-Pipeline/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data_resp = json.loads(resp.read().decode("utf-8"))
            candidates = data_resp.get("candidates", [])
            if candidates and "content" in candidates[0] and "parts" in candidates[0]["content"]:
                parts = candidates[0]["content"]["parts"]
                if parts and "text" in parts[0]:
                    return parts[0]["text"].strip()
            raise ValueError(f"Unexpected Google API response format: {data_resp}")

    def _call_openai_compatible(self, prompt: str, temperature: float, max_tokens: int, top_p: float) -> str:
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "FactIndic-Demo-Pipeline/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data_resp = json.loads(resp.read().decode("utf-8"))
            choices = data_resp.get("choices", [])
            if choices and "message" in choices[0] and "content" in choices[0]["message"]:
                return choices[0]["message"]["content"].strip()
            raise ValueError(f"Unexpected OpenAI-compatible response format: {data_resp}")

    def _call_cohere(self, prompt: str, temperature: float, max_tokens: int, top_p: float) -> str:
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "p": top_p,
            "thinking": {"type": "disabled"},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "FactIndic-Demo-Pipeline/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data_resp = json.loads(resp.read().decode("utf-8"))
            content = data_resp.get("message", {}).get("content", [])
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
            raise ValueError(f"Unexpected Cohere response format: {data_resp}")

    def _call_huggingface(self, prompt: str, temperature: float, max_tokens: int, top_p: float) -> str:
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "top_p": top_p,
                "return_full_text": False,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "FactIndic-Demo-Pipeline/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data_resp = json.loads(resp.read().decode("utf-8"))
            if isinstance(data_resp, list) and len(data_resp) > 0 and "generated_text" in data_resp[0]:
                return data_resp[0]["generated_text"].strip()
            raise ValueError(f"Unexpected Hugging Face response format: {data_resp}")
