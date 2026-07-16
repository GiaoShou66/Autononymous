from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

import requests


# Default endpoints
OLLAMA_BASE_URL = "http://localhost:11434"
OPENAI_BASE_URL = "https://api.openai.com/v1"

# Backend detection helpers
def _is_openai_model(model_name: str) -> bool:
    """Return True if model_name looks like an OpenAI/compatible model."""
    openai_prefixes = ("gpt-", "o1", "o3", "o4", "text-davinci", "chatgpt")
    return any(model_name.lower().startswith(p) for p in openai_prefixes)


def _is_custom_model(model_name: str) -> bool:
    """Return True if model_name starts with 'custom:' prefix."""
    return model_name.lower().startswith("custom:")

# Ollama backend
@dataclass
class _OllamaBackend:
    model_name: str
    base_url: str = OLLAMA_BASE_URL

    def _chat_endpoint(self) -> str:
        return f"{self.base_url}/api/chat"

    def _tags_endpoint(self) -> str:
        return f"{self.base_url}/api/tags"

    def _installed_models(self) -> List[str]:
        resp = requests.get(self._tags_endpoint(), timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models") or []
        return [m["name"] for m in models if m.get("name")]

    def _resolve_model_name(self) -> str:
        """
        Return the configured model if installed, otherwise fall back to a
        close local match.
        """
        try:
            installed = self._installed_models()
        except Exception:
            return self.model_name

        if not installed:
            return self.model_name
        if self.model_name in installed:
            return self.model_name

        configured_base = self.model_name.split(":", 1)[0]
        for name in installed:
            if name.split(":", 1)[0] == configured_base:
                return name

        return installed[0]

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        model_name = self._resolve_model_name()
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        try:
            resp = requests.post(self._chat_endpoint(), json=payload, timeout=300)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            details = str(exc)
            if exc.response is not None and exc.response.text:
                details += f"\nBackend response: {exc.response.text[:1000]}"
            # Retry without tools if model doesn't support them
            if tools and "does not support tools" in details:
                payload.pop("tools", None)
                try:
                    resp = requests.post(self._chat_endpoint(), json=payload, timeout=300)
                    resp.raise_for_status()
                    data = resp.json()
                    return data.get("message") or {"role": "assistant", "content": ""}
                except Exception as fallback_exc:
                    details += f"\nFallback without tools failed: {fallback_exc}"
            return {"role": "assistant", "content": f"[Ollama error] {details}"}
        except Exception as exc:
            return {"role": "assistant", "content": f"[Ollama error] {exc}"}

        data = resp.json()
        return data.get("message") or {"role": "assistant", "content": ""}


# OpenAI-compatible backend (also works with Together, Groq, Mistral, etc.)
@dataclass
class _OpenAIBackend:
    model_name: str
    base_url: str = OPENAI_BASE_URL
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    def _chat_endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "role": "assistant",
                "content": (
                    "[OpenAI backend] No API key found. "
                    "Set the OPENAI_API_KEY environment variable."
                ),
            }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools

        try:
            resp = requests.post(
                self._chat_endpoint(), json=payload, headers=headers, timeout=300
            )
            resp.raise_for_status()
        except requests.HTTPError as exc:
            details = str(exc)
            if exc.response is not None and exc.response.text:
                details += f"\nBackend response: {exc.response.text[:1000]}"
            return {"role": "assistant", "content": f"[OpenAI error] {details}"}
        except Exception as exc:
            return {"role": "assistant", "content": f"[OpenAI error] {exc}"}

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return {"role": "assistant", "content": ""}

        msg = choices[0].get("message", {})
        # Normalise tool_calls to Ollama-style format
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            normalised = []
            for tc in tool_calls:
                normalised.append({
                    "function": {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", {}),
                    }
                })
            return {
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": normalised,
            }
        return {"role": "assistant", "content": msg.get("content") or ""}


@dataclass
class _CustomBackend:
    """
    Generic backend for any OpenAI-compatible or custom endpoint.

    In config.yaml set:
      model_name: "custom:my-model-name"

    Then set environment variables:
      CUSTOM_MODEL_URL  — full chat completions URL (required)
      CUSTOM_MODEL_KEY  — API key / bearer token (optional)
    """
    model_name: str  # without the "custom:" prefix
    endpoint: str = field(default_factory=lambda: os.getenv("CUSTOM_MODEL_URL", ""))
    api_key: str = field(default_factory=lambda: os.getenv("CUSTOM_MODEL_KEY", ""))

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        if not self.endpoint:
            return {
                "role": "assistant",
                "content": (
                    "[Custom backend] No endpoint configured. "
                    "Set the CUSTOM_MODEL_URL environment variable to your "
                    "chat completions URL (e.g. http://localhost:8000/v1/chat/completions)."
                ),
            }

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools

        try:
            resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=300)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            details = str(exc)
            if exc.response is not None and exc.response.text:
                details += f"\nBackend response: {exc.response.text[:1000]}"
            return {"role": "assistant", "content": f"[Custom error] {details}"}
        except Exception as exc:
            return {"role": "assistant", "content": f"[Custom error] {exc}"}

        data = resp.json()
        # Support both Ollama-style {"message": {...}} and OpenAI-style {"choices": [...]}
        if "message" in data:
            return data["message"] or {"role": "assistant", "content": ""}
        choices = data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                normalised = [
                    {
                        "function": {
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": tc.get("function", {}).get("arguments", {}),
                        }
                    }
                    for tc in tool_calls
                ]
                return {
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": normalised,
                }
            return {"role": "assistant", "content": msg.get("content") or ""}
        return {"role": "assistant", "content": ""}

# Unified ModelClient (auto-selects backend)
@dataclass
class ModelClient:
    """
    Unified model client that supports multiple backends:

      Ollama (local)        — default, model_name like "llama3:latest"
      OpenAI-compatible     — model_name like "gpt-4o", "gpt-3.5-turbo"
      Custom / self-hosted  — model_name like "custom:my-model"
                              requires CUSTOM_MODEL_URL env var

    Environment variables:
      OPENAI_API_KEY   — for OpenAI / OpenAI-compatible APIs
      CUSTOM_MODEL_URL — full chat completions URL for custom backend
      CUSTOM_MODEL_KEY — optional auth token for custom backend
      OLLAMA_BASE_URL  — override Ollama URL (default: http://localhost:11434)
    """

    model_name: str
    base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL))

    def _get_backend(self):
        if _is_custom_model(self.model_name):
            real_name = self.model_name.split(":", 1)[1] if ":" in self.model_name else self.model_name
            return _CustomBackend(model_name=real_name)
        if _is_openai_model(self.model_name):
            return _OpenAIBackend(model_name=self.model_name, base_url=OPENAI_BASE_URL)
        return _OllamaBackend(model_name=self.model_name, base_url=self.base_url)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """
        Call the selected backend and return a normalised assistant message dict.
        """
        return self._get_backend().chat(messages, tools=tools)

    def generate_response(self, messages: List[Dict[str, Any]]) -> str:
        """
        Simple wrapper that returns just the content string (no tool support).
        """
        msg = self.chat(messages)
        return msg.get("content") or ""
