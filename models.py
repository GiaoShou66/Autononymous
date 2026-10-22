from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Callable

import requests
from requests.exceptions import ChunkedEncodingError


# Default endpoints
OLLAMA_BASE_URL = "http://localhost:11434"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"

_COMPLEXITY_HINTS = (
    "analyze", "architecture", "debug", "design", "diagnose", "explain why",
    "implement", "investigate", "optimize", "plan", "reason", "refactor",
    "review", "security", "step by step", "test", "tradeoff",
)


def _stream_ollama_response(response, on_delta: Callable[[str], None]) -> Dict[str, Any]:
    """Consume Ollama's newline-delimited streaming chat response."""
    content: list[str] = []
    tool_calls = None
    try:
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            try:
                chunk = json.loads(line)
                message = chunk.get("message") or {}
                text = message.get("content") or ""
                if text:
                    content.append(text)
                    try:
                        on_delta(text)
                    except Exception as delta_err:
                        print(f"Warning: on_delta callback error: {delta_err}")
                if message.get("tool_calls"):
                    tool_calls = message["tool_calls"]
                if chunk.get("done"):
                    break
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {line[:100]}")
                continue
    except ChunkedEncodingError as e:
        print(f"Stream chunked encoding error: {e}")
    except Exception as e:
        print(f"Stream error: {e}")
        if not content:
            content = ["(Stream interrupted)"]
    
    result = {"role": "assistant", "content": "".join(content)}
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def _stream_chat_completions_response(response, on_delta: Callable[[str], None]) -> Dict[str, Any]:
    """Consume a chat/completions-style SSE stream, including streamed function calls."""
    content: list[str] = []
    tool_calls: dict[int, Dict[str, Any]] = {}
    full_message: Dict[str, Any] | None = None

    try:
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            line = line.strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                break
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Accept a server that advertises streaming but sends Ollama-style chunks.
            if isinstance(event.get("message"), dict):
                message = event["message"]
                text = message.get("content") or ""
                if text:
                    content.append(text)
                    on_delta(text)
                if message.get("tool_calls"):
                    full_message = message
                if event.get("done"):
                    break
                continue

            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            if choice.get("message"):
                full_message = choice["message"]
                text = full_message.get("content") or ""
                if text:
                    content.append(text)
                    on_delta(text)
                for index, call in enumerate(full_message.get("tool_calls") or []):
                    tool_calls[index] = call
            text = delta.get("content") or ""
            if isinstance(text, str) and text:
                content.append(text)
                try:
                    on_delta(text)
                except Exception as delta_err:
                    print(f"Warning: on_delta callback error: {delta_err}")
            for call in delta.get("tool_calls") or []:
                index = int(call.get("index", 0))
                target = tool_calls.setdefault(index, {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                })
                if call.get("id"):
                    target["id"] = call["id"]
                function = call.get("function") or {}
                if function.get("name"):
                    target["function"]["name"] += function["name"]
                if function.get("arguments"):
                    target["function"]["arguments"] += function["arguments"]
    except ChunkedEncodingError as e:
        print(f"Stream chunked encoding error: {e}")
    except Exception as e:
        print(f"Stream error: {e}")
        if not content:
            content = ["(Stream interrupted)"]

    result = {"role": "assistant", "content": "".join(content)}
    if full_message:
        result["content"] = full_message.get("content") or result["content"]
        calls = full_message.get("tool_calls") or list(tool_calls.values())
    else:
        calls = [tool_calls[index] for index in sorted(tool_calls)]
    if calls:
        def _normalise_call(call):
            # function.arguments must always be a JSON-encoded string —
            # calls built incrementally above already satisfy this, but
            # calls taken straight from full_message.get("tool_calls")
            # (an unnormalized, backend-provided object) might not.
            raw_arguments = (call.get("function") or {}).get("arguments", "{}")
            arguments = raw_arguments if isinstance(raw_arguments, str) else json.dumps(raw_arguments)
            return {
                "id": call.get("id"),
                # .get("type", "function") would only catch a missing key —
                # some backends (e.g. CommandCode AI) return "type": null
                # explicitly, which .get() happily passes through as None.
                # `or` catches that too; "function" is the only valid value.
                "type": call.get("type") or "function",
                "function": {
                    "name": (call.get("function") or {}).get("name", ""),
                    "arguments": arguments,
                },
            }
        result["tool_calls"] = [_normalise_call(call) for call in calls]
    return result

# Backend detection helpers
def _is_anthropic_model(model_name: str) -> bool:
    """Return True if model_name looks like an Anthropic model."""
    normalized = model_name.strip().lower()
    anthropic_prefixes = ("claude-", "claude")
    return any(normalized.startswith(p) for p in anthropic_prefixes)


# Ollama backend
@dataclass
class _OllamaBackend:
    model_name: str
    base_url: str = OLLAMA_BASE_URL
    timeout: int = 300
    reasoning: bool = False

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
        configured_suffix = configured_base.split("/")[-1]

        # 1. Exact base match (ignoring tag)
        for name in installed:
            base = name.split(":", 1)[0]
            if base == configured_base:
                return name

        # 2. Match without namespace (e.g. "claude-mythos-5-1" matches "treyleo16/claude-mythos-5-1:latest")
        for name in installed:
            base = name.split(":", 1)[0]
            if base.split("/")[-1] == configured_suffix:
                return name

        return self.model_name

    def _truncate_messages(self, messages: List[Dict[str, Any]], max_chars: int = 60000) -> List[Dict[str, Any]]:
        """Truncate messages to fit within a character limit to prevent Ollama truncation errors."""
        current_total = sum(len(msg.get("content", "")) for msg in messages if isinstance(msg.get("content"), str))
        if current_total <= max_chars:
            return messages

        # Truncate oldest messages first, keeping the most recent ones and the system prompt
        truncated = []
        system_prompt = None
        
        # Separate system prompt
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg
            else:
                truncated.append(msg)
        
        # Keep the most recent messages that fit
        final_messages = []
        running_total = 0
        for msg in reversed(truncated):
            content_len = len(msg.get("content", "")) if isinstance(msg.get("content"), str) else 0
            if running_total + content_len > max_chars:
                break
            final_messages.append(msg)
            running_total += content_len
        
        final_messages.reverse()
        if system_prompt:
            final_messages.insert(0, system_prompt)
            
        return final_messages

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> Dict[str, Any]:
        model_name = self._resolve_model_name()
        
        # Prevent Ollama "closing '}' symbol" error by limiting payload size
        messages = self._truncate_messages(messages)
        
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": on_delta is not None,
        }
        # Ollama's native thinking mode is typically handled by the model
        # internally or via prompt templates, not a top-level API parameter.
        # Removed payload["think"] = True as it causes 400 Bad Request.
        num_predict = os.getenv("OLLAMA_NUM_PREDICT")
        if num_predict and num_predict.isdigit() and int(num_predict) > 0:
            payload["options"] = {"num_predict": int(num_predict)}
        if tools:
            payload["tools"] = tools

        try:
            # Use explicit json.dumps for better control and logging
            json_payload = json.dumps(payload)
            # Log payload size for debugging truncation issues
            # print(f"Sending Ollama request. Size: {len(json_payload)} bytes. Model: {model_name}")
            
            resp = requests.post(
                self._chat_endpoint(), 
                data=json_payload, 
                headers={'Content-Type': 'application/json'}, 
                timeout=self.timeout,
                stream=on_delta is not None,
            )
            resp.raise_for_status()
        except requests.HTTPError as exc:
            details = str(exc)
            if exc.response is not None and exc.response.text:
                details += f"\nBackend response: {exc.response.text[:1000]}"
            # Retry without tools if model doesn't support them
            if tools and "does not support tools" in details:
                payload.pop("tools", None)
                try:
                    resp = requests.post(
                        self._chat_endpoint(), json=payload, timeout=self.timeout,
                        stream=on_delta is not None,
                    )
                    resp.raise_for_status()
                    if on_delta:
                        try:
                            return _stream_ollama_response(resp, on_delta)
                        finally:
                            resp.close()
                    data = resp.json()
                    return data.get("message") or {"role": "assistant", "content": ""}
                except Exception as fallback_exc:
                    details += f"\nFallback without tools failed: {fallback_exc}"
            return {"role": "assistant", "content": f"[Ollama error] {details}"}
        except Exception as exc:
            return {"role": "assistant", "content": f"[Ollama error] {exc}"}

        if on_delta:
            try:
                return _stream_ollama_response(resp, on_delta)
            finally:
                resp.close()
        data = resp.json()
        return data.get("message") or {"role": "assistant", "content": ""}


# Generic chat/completions backend (works with CommandCode AI, Together,
# Groq, Mistral, and any other gateway speaking the same wire protocol).
@dataclass
class _ChatCompletionsBackend:
    """
    Generic client for any backend speaking the widely-used chat/completions
    wire protocol. This is the shared HTTP engine for providers that speak
    that protocol: CommandCode AI (_CommandCodeBackend below) and the
    custom_model_url escape hatch in _get_backend. It calls only the
    base_url it is configured with — never any fixed third-party vendor API.
    """
    model_name: str
    base_url: str
    api_key: str = ""
    timeout: int = 300
    label: str = "Chat-completions"

    def _chat_endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def _message_from_chat_completions_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        choices = data.get("choices", [])
        if not choices:
            return {"role": "assistant", "content": ""}

        msg = choices[0].get("message", {})
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            normalised = []
            for tc in tool_calls:
                # function.arguments must always be a JSON-encoded string
                # per the OpenAI-style spec. Some backends (e.g. CommandCode
                # AI) hand back an already-parsed object instead, or the
                # key is missing entirely (default {} is itself a dict,
                # not a string) — either way, re-encode anything that
                # isn't already a string, or the backend rejects it on the
                # next turn with "expected string, received object".
                raw_arguments = tc.get("function", {}).get("arguments", "{}")
                arguments = raw_arguments if isinstance(raw_arguments, str) else json.dumps(raw_arguments)
                normalised.append({
                    "id": tc.get("id"),
                    # .get("type", "function") only catches a missing key —
                    # some backends (e.g. CommandCode AI) return
                    # "type": null explicitly, which .get() passes through
                    # as None. `or` catches that too.
                    "type": tc.get("type") or "function",
                    "function": {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": arguments,
                    }
                })
            return {
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": normalised,
            }
        return {"role": "assistant", "content": msg.get("content") or ""}

    def _post_chat(self, payload: Dict[str, Any], headers: Dict[str, str], stream: bool):
        try:
            resp = requests.post(
                self._chat_endpoint(), json=payload, headers=headers, timeout=self.timeout,
                stream=stream,
            )
            resp.raise_for_status()
            return resp, None
        except requests.HTTPError as exc:
            details = str(exc)
            if exc.response is not None and exc.response.text:
                details += f"\nBackend response: {exc.response.text[:1000]}"
            status = exc.response.status_code if exc.response is not None else None
            return None, {
                "role": "assistant",
                "content": f"[{self.label} error] {details}",
                "_status": status,
            }
        except Exception as exc:
            return None, {"role": "assistant", "content": f"[{self.label} error] {exc}"}

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "role": "assistant",
                "content": f"[{self.label}] No API key found for this backend.",
            }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": on_delta is not None,
        }
        if tools:
            payload["tools"] = tools

        resp, error = self._post_chat(payload, headers, stream=on_delta is not None)
        if error:
            status = error.pop("_status", None)
            # Some gateway models reject the "tools" param outright (400)
            # instead of reporting "does not support tools" the way Ollama
            # does. Retry once without tools before surfacing the error.
            if status == 400 and tools:
                retry_payload = dict(payload)
                retry_payload.pop("tools", None)
                resp, retry_error = self._post_chat(retry_payload, headers, stream=on_delta is not None)
                if retry_error:
                    retry_error.pop("_status", None)
                    return error
                payload = retry_payload
            else:
                return error

        if on_delta:
            try:
                streamed = _stream_chat_completions_response(resp, on_delta)
            except ChunkedEncodingError as exc:
                print(f"Stream chunked encoding error: {exc}")
                streamed = {"role": "assistant", "content": ""}
            finally:
                resp.close()

            content = (streamed.get("content") or "").strip()
            interrupted = content.startswith("(Stream interrupted")
            if streamed.get("tool_calls") or (content and not interrupted):
                return streamed

            retry_payload = dict(payload)
            retry_payload["stream"] = False
            retry, retry_error = self._post_chat(retry_payload, headers, stream=False)
            if retry_error:
                return retry_error
            try:
                return self._message_from_chat_completions_json(retry.json())
            except Exception as exc:
                return {"role": "assistant", "content": f"[{self.label} error] {exc}"}
            finally:
                retry.close()

        return self._message_from_chat_completions_json(resp.json())


def _is_commandcode_model(model_name: str) -> bool:
    """Return True if model_name starts with the 'commandcode:' prefix."""
    return model_name.lower().startswith("commandcode:")


@dataclass
class _CommandCodeBackend(_ChatCompletionsBackend):
    """
    CommandCode AI (api.commandcode.ai) — a chat/completions-compatible
    gateway in front of many providers (DeepSeek, Kimi, GPT, GLM, Qwen,
    Gemini, Grok, Mistral, ...). The base URL is a fixed default here,
    confirmed live: POST .../chat/completions returns 401 (not 404) without
    a key, so the route is correct.

    Trigger with model_name "commandcode:<model-id>", e.g.
    "commandcode:deepseek/deepseek-v4-pro". List valid ids with
    GET https://api.commandcode.ai/provider/v1/models (no auth required).

    Set COMMANDCODE_API_KEY as an environment variable — never put a real
    key in config.yaml, which is git-tracked.

    Note: this gateway's claude-* model ids only support /messages
    (Anthropic's native format), not /chat/completions, so they will not
    work through this backend.
    """
    base_url: str = "https://api.commandcode.ai/provider/v1"
    api_key: str = field(default_factory=lambda: os.getenv("COMMANDCODE_API_KEY", ""))
    label: str = "CommandCode backend"


def _is_openrouter_model(model_name: str) -> bool:
    """Return True if model_name starts with the 'openrouter:' prefix."""
    return model_name.lower().startswith("openrouter:")


@dataclass
class _OpenRouterBackend(_ChatCompletionsBackend):
    """
    OpenRouter (openrouter.ai) — a chat/completions-compatible gateway in
    front of hundreds of models. Confirmed live: GET /v1/models works with
    no auth (470 models at last check); POST /v1/chat/completions returns
    401 (not 404) without a key, so the route is correct.

    Trigger with model_name "openrouter:<model-id>", e.g.
    "openrouter:anthropic/claude-3.5-sonnet" or "openrouter:openai/gpt-4o".
    List valid ids with GET https://openrouter.ai/api/v1/models (no auth).

    Set OPENROUTER_API_KEY as an environment variable — never put a real
    key in config.yaml, which is git-tracked.
    """
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    label: str = "OpenRouter backend"


def _is_deepseek_model(model_name: str) -> bool:
    """Return True if model_name starts with the 'deepseek:' prefix."""
    return model_name.lower().startswith("deepseek:")


@dataclass
class _DeepSeekBackend(_ChatCompletionsBackend):
    """
    DeepSeek's own API (api.deepseek.com) — direct access, not through a
    gateway. Confirmed live: POST /v1/chat/completions returns 401 (not
    404) without a key, so the route is correct. (GET /v1/models also
    requires auth on this API, unlike OpenRouter's public one.)

    Trigger with model_name "deepseek:<model-id>", e.g. "deepseek:deepseek-chat"
    or "deepseek:deepseek-reasoner".

    Set DEEPSEEK_API_KEY as an environment variable — never put a real key
    in config.yaml, which is git-tracked.
    """
    base_url: str = "https://api.deepseek.com/v1"
    api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    label: str = "DeepSeek backend"


def _is_lmarena_model(model_name: str) -> bool:
    """Return True if model_name starts with the 'lmarena:' prefix."""
    return model_name.lower().startswith("lmarena:")


@dataclass
class _LMArenaBackend(_ChatCompletionsBackend):
    """
    lmarena2api (https://github.com/deanxv/lmarena2api) — a self-hosted,
    OpenAI-compatible proxy in front of the chat models exposed on
    lmarena.ai (Claude, GPT, Gemini, Grok, etc. — whatever LMArena
    currently lists). Unlike the other backends above this is not a fixed
    third-party vendor API: you run the lmarena2api server yourself, so
    base_url is configurable rather than hardcoded to a public domain.

    Trigger with model_name "lmarena:<model-id>", e.g.
    "lmarena:claude-3-5-sonnet-20241022" or "lmarena:chatgpt-4o-latest" —
    use whatever model id lmarena2api's README currently lists as
    supported for that server instance.

    Set LMARENA_BASE_URL to your running instance's API root (default:
    http://localhost:10088/v1, lmarena2api's default port) and
    LMARENA_API_KEY to the API_SECRET you set when starting the
    lmarena2api container — never put a real key in config.yaml, which is
    git-tracked.
    """
    base_url: str = field(
        default_factory=lambda: os.getenv("LMARENA_BASE_URL", "http://localhost:10088/v1")
    )
    api_key: str = field(default_factory=lambda: os.getenv("LMARENA_API_KEY", ""))
    label: str = "lmarena2api backend"


@dataclass
class _AnthropicBackend:
    """
    Anthropic Claude API backend.
    
    Supports Claude models: claude-3-opus, claude-3-sonnet, claude-3-haiku,
    claude-3-5-sonnet, etc.
    
    Environment variable:
      ANTHROPIC_API_KEY — required for authentication
    """
    model_name: str
    base_url: str = ANTHROPIC_BASE_URL
    api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    timeout: int = 300
    
    def _messages_endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/messages"
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "role": "assistant",
                "content": (
                    "[Anthropic backend] No API key found. "
                    "Set the ANTHROPIC_API_KEY environment variable."
                ),
            }
        
        # Convert messages to Anthropic format
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role == "system":
                system_message = content
            elif role in ["user", "assistant"]:
                anthropic_messages.append({
                    "role": role,
                    "content": content
                })
            elif role == "tool":
                # Anthropic uses tool_result
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": content
                    }]
                })
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": anthropic_messages,
            "max_tokens": 4096,
            "stream": on_delta is not None,
        }
        
        if system_message:
            payload["system"] = system_message
        
        # Convert tools to Anthropic format
        if tools:
            anthropic_tools = []
            for tool in tools:
                func = tool.get("function", {})
                anthropic_tools.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {})
                })
            payload["tools"] = anthropic_tools
        
        try:
            resp = requests.post(
                self._messages_endpoint(),
                json=payload,
                headers=headers,
                timeout=self.timeout,
                stream=on_delta is not None,
            )
            resp.raise_for_status()
        except requests.HTTPError as exc:
            details = str(exc)
            if exc.response is not None and exc.response.text:
                details += f"\nBackend response: {exc.response.text[:1000]}"
            return {"role": "assistant", "content": f"[Anthropic error] {details}"}
        except Exception as exc:
            return {"role": "assistant", "content": f"[Anthropic error] {exc}"}
        
        if on_delta:
            # Handle streaming
            try:
                return self._stream_anthropic_response(resp, on_delta)
            finally:
                resp.close()
        
        # Parse non-streaming response
        data = resp.json()
        
        # Extract content
        content_blocks = data.get("content", [])
        text_content = ""
        tool_calls = []
        
        for block in content_blocks:
            if block.get("type") == "text":
                text_content += block.get("text", "")
            elif block.get("type") == "tool_use":
                # Convert to chat-completions-style tool call
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}))
                    }
                })
        
        result = {
            "role": "assistant",
            "content": text_content,
        }
        
        if tool_calls:
            result["tool_calls"] = tool_calls
        
        return result
    
    def _stream_anthropic_response(
        self,
        response,
        on_delta: Callable[[str], None]
    ) -> Dict[str, Any]:
        """Handle Anthropic streaming response"""
        content = []
        tool_calls = []
        
        try:
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                line = line.strip()
                
                if not line.startswith("data: "):
                    continue
                
                data_str = line[6:]  # Remove "data: " prefix
                
                if data_str == "[DONE]":
                    break
                
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                
                event_type = event.get("type")
                
                if event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            content.append(text)
                            try:
                                on_delta(text)
                            except Exception as delta_err:
                                print(f"Warning: on_delta callback error: {delta_err}")
                
                elif event_type == "content_block_start":
                    block = event.get("content_block", {})
                    if block.get("type") == "tool_use":
                        tool_calls.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": ""
                            }
                        })
        except Exception as e:
            print(f"Anthropic stream error: {e}")
            if not content:
                content = ["(Stream interrupted)"]
        
        result = {
            "role": "assistant",
            "content": "".join(content)
        }
        
        if tool_calls:
            result["tool_calls"] = tool_calls
        
        return result


# Unified ModelClient (auto-selects backend)
@dataclass
class ModelClient:
    """
    Unified model client that supports multiple backends:

      Ollama (local)        — default, model_name like "llama3:latest"
      Anthropic Claude      — model_name like "claude-3-5-sonnet-20241022"
      CommandCode AI        — model_name like "commandcode:deepseek/deepseek-v4-pro"
      OpenRouter            — model_name like "openrouter:anthropic/claude-3.5-sonnet"
      DeepSeek (direct)     — model_name like "deepseek:deepseek-chat"
      lmarena2api (self-hosted) — model_name like "lmarena:claude-3-5-sonnet-20241022"
      Custom endpoint       — set custom_model_url; routes any model_name there

    This client never calls any fixed third-party vendor API outside the
    backends listed above. Custom endpoints (and lmarena2api, which you run
    yourself) merely speak the same widely-used chat/completions wire
    protocol that CommandCode AI and others also use.

    Environment variables:
      ANTHROPIC_API_KEY   — for Anthropic Claude models
      COMMANDCODE_API_KEY — for CommandCode AI models
      OPENROUTER_API_KEY  — for OpenRouter models
      DEEPSEEK_API_KEY    — for DeepSeek's own API (direct, not via a gateway)
      LMARENA_BASE_URL    — your lmarena2api instance (default: http://localhost:10088/v1)
      LMARENA_API_KEY     — the API_SECRET configured on that lmarena2api instance
      CUSTOM_MODEL_KEY    — for the custom_model_url backend, if it needs auth
      OLLAMA_BASE_URL     — override Ollama URL (default: http://localhost:11434)
      OLLAMA_TIMEOUT      — override request timeout in seconds (default: 300)
    """

    model_name: str
    base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL))
    timeout: int = field(default_factory=lambda: int(os.getenv("OLLAMA_TIMEOUT", "300")))
    reasoning: bool = field(
        default_factory=lambda: os.getenv("AI_AGENT_REASONING", "true").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    routing_mode: str = "single"
    fast_model: str = ""
    reasoning_model: str = ""
    custom_model_url: str = ""
    custom_model_key: str = ""
    last_routed_model: str = field(default="", init=False)

    def _route_model(self, messages: List[Dict[str, Any]]) -> str:
        """
        Pick a fast or reasoning model without an extra classifier request.
        Used directly by hybrid mode (whichever model this returns handles
        the whole turn alone) and by relay mode (as the complexity check
        deciding whether to run the reasoning-then-fast relay for this
        turn, or just use the fast model like a simple hybrid turn would).
        """
        if self.routing_mode.strip().lower() not in {"hybrid", "relay"}:
            self.last_routed_model = self.model_name
            return self.model_name

        user_text = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                user_text = content if isinstance(content, str) else json.dumps(content)
                break

        normalized = user_text.lower()
        complexity = sum(hint in normalized for hint in _COMPLEXITY_HINTS)
        if len(user_text) >= 800:
            complexity += 1
        if user_text.count("\n") >= 4 or "```" in user_text:
            complexity += 1

        selected = (
            self.reasoning_model.strip()
            if complexity >= 2 and self.reasoning_model.strip()
            else self.fast_model.strip()
        )
        self.last_routed_model = selected or self.model_name
        return self.last_routed_model

    def routing_context(self) -> str:
        """Return factual routing state for inclusion in the agent prompt."""
        mode = self.routing_mode.strip().lower()
        if mode == "relay":
            return (
                "Model routing: relay mode is active. For a turn judged complex, the "
                "reasoning model (" + (self.reasoning_model or self.model_name) + ") first "
                "produces an internal text-only analysis (no tools), which is handed to the "
                "fast model (" + (self.fast_model or self.model_name) + ") as extra context; "
                "the fast model then produces the actual reply, including any tool calls. "
                "Simple turns skip the reasoning pass and use the fast model alone. "
                "Last model to produce a visible reply: "
                + (self.last_routed_model or "not selected yet") + ". "
                "When asked about routing, use only these values; do not invent hardware, "
                "cloud, or model-installation details."
            )
        if mode != "hybrid":
            return "Model routing: single-model mode using " + self.model_name + "."
        return (
            "Model routing: hybrid mode is active. Each request is handled by exactly one "
            "model. Fast model: " + (self.fast_model or self.model_name) + ". "
            "Reasoning model: " + (self.reasoning_model or self.model_name) + ". "
            "Last selected model: " + (self.last_routed_model or "not selected yet") + ". "
            "When asked about routing, use only these values; do not invent hardware, cloud, "
            "or model-installation details."
        )

    @staticmethod
    def _is_missing_model_error(response: Dict[str, Any]) -> bool:
        content = str(response.get("content") or "").lower()
        return content.startswith("[ollama error]") and "model" in content and "not found" in content

    def _chat_routed(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> Dict[str, Any]:
        selected_model = self._route_model(messages)
        backend = self._get_backend(selected_model)
        response = backend.chat(messages, tools=tools, on_delta=on_delta)
        if (
            selected_model != self.model_name
            and isinstance(backend, _OllamaBackend)
            and self._is_missing_model_error(response)
        ):
            self.last_routed_model = self.model_name
            response = self._get_backend(self.model_name).chat(
                messages, tools=tools, on_delta=on_delta
            )
        return response

    def _chat_relay(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> Dict[str, Any]:
        """
        routing_mode "relay": for a turn _route_model judges complex, the
        reasoning model drafts a text-only analysis first (no tools — a
        scratchpad pass, not a tool-executing turn), then the fast model
        gets that draft injected as extra context and produces the actual
        reply (including any tool calls). Simple turns skip the draft pass
        entirely and just use the fast model, same as hybrid mode would.
        """
        selected_model = self._route_model(messages)
        reasoning_model = self.reasoning_model.strip()

        if selected_model != reasoning_model or not reasoning_model:
            # Simple turn (or no reasoning model configured) -> behaves
            # exactly like hybrid mode's non-complex path.
            backend = self._get_backend(selected_model)
            response = backend.chat(messages, tools=tools, on_delta=on_delta)
            if (
                selected_model != self.model_name
                and isinstance(backend, _OllamaBackend)
                and self._is_missing_model_error(response)
            ):
                self.last_routed_model = self.model_name
                response = self._get_backend(self.model_name).chat(
                    messages, tools=tools, on_delta=on_delta
                )
            return response

        reasoning_backend = self._get_backend(reasoning_model)
        draft = reasoning_backend.chat(messages, tools=None, on_delta=None)
        draft_text = (draft.get("content") or "").strip()

        fast_model = self.fast_model.strip() or self.model_name
        augmented = list(messages)
        if draft_text:
            augmented.append({
                "role": "system",
                "content": (
                    "Internal analysis from a stronger reasoning model, for your "
                    "own use only — do not mention this note or that a separate "
                    "model produced it; just use it to inform your reply:\n\n"
                    + draft_text
                ),
            })

        self.last_routed_model = fast_model
        fast_backend = self._get_backend(fast_model)
        response = fast_backend.chat(augmented, tools=tools, on_delta=on_delta)
        if (
            fast_model != self.model_name
            and isinstance(fast_backend, _OllamaBackend)
            and self._is_missing_model_error(response)
        ):
            self.last_routed_model = self.model_name
            response = self._get_backend(self.model_name).chat(
                augmented, tools=tools, on_delta=on_delta
            )
        return response

    def _get_backend(self, model_name: str | None = None):
        selected_model = model_name or self.model_name
        if self.custom_model_url:
            return _ChatCompletionsBackend(
                model_name=selected_model,
                base_url=self.custom_model_url,
                api_key=self.custom_model_key or os.getenv("CUSTOM_MODEL_KEY", ""),
                timeout=self.timeout,
                label="Custom backend",
            )
        if _is_commandcode_model(selected_model):
            real_name = selected_model.split(":", 1)[1] if ":" in selected_model else selected_model
            return _CommandCodeBackend(model_name=real_name, timeout=self.timeout)
        if _is_openrouter_model(selected_model):
            real_name = selected_model.split(":", 1)[1] if ":" in selected_model else selected_model
            return _OpenRouterBackend(model_name=real_name, timeout=self.timeout)
        if _is_deepseek_model(selected_model):
            real_name = selected_model.split(":", 1)[1] if ":" in selected_model else selected_model
            return _DeepSeekBackend(model_name=real_name, timeout=self.timeout)
        if _is_lmarena_model(selected_model):
            real_name = selected_model.split(":", 1)[1] if ":" in selected_model else selected_model
            return _LMArenaBackend(model_name=real_name, timeout=self.timeout)
        if _is_anthropic_model(selected_model):
            return _AnthropicBackend(model_name=selected_model, base_url=ANTHROPIC_BASE_URL, timeout=self.timeout)
        return _OllamaBackend(
            model_name=selected_model,
            base_url=self.base_url,
            timeout=self.timeout,
            reasoning=self.reasoning,
        )

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """
        Call the selected backend and return a normalised assistant message dict.
        """
        if self.routing_mode.strip().lower() == "relay":
            return self._chat_relay(messages, tools=tools)
        return self._chat_routed(messages, tools=tools)

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> Dict[str, Any]:
        """Stream visible text while returning a complete normalized message."""
        if self.routing_mode.strip().lower() == "relay":
            return self._chat_relay(messages, tools=tools, on_delta=on_delta)
        return self._chat_routed(messages, tools=tools, on_delta=on_delta)

    def list_available_models(self) -> List[str]:
        """List model IDs: local Ollama models + provider models."""
        models: List[str] = []
        try:
            ollama = _OllamaBackend(model_name="list", base_url=self.base_url)
            installed = ollama._installed_models()
            if installed:
                models.extend(installed)
        except Exception:
            pass

        try:
            resp = requests.get("https://api.commandcode.ai/provider/v1/models", timeout=4)
            if resp.status_code == 200:
                for item in resp.json().get("data", []):
                    m_id = item.get("id")
                    if m_id:
                        models.append(f"commandcode:{m_id}")
        except Exception:
            pass

        try:
            resp = requests.get("https://openrouter.ai/api/v1/models", timeout=4)
            if resp.status_code == 200:
                for item in resp.json().get("data", []):
                    m_id = item.get("id")
                    if m_id:
                        models.append(f"openrouter:{m_id}")
        except Exception:
            pass

        # DeepSeek's own /models endpoint requires a valid key even to list
        # (confirmed: 401 with no auth at all), so it can't be browsed
        # publicly like OpenRouter's or CommandCode's. These two are
        # DeepSeek's documented, stable model ids.
        models.append("deepseek:deepseek-chat")
        models.append("deepseek:deepseek-reasoner")

        if self.model_name and self.model_name not in models:
            models.insert(0, self.model_name)
        return models or [self.model_name]

    def generate_response(self, messages: List[Dict[str, Any]]) -> str:
        """
        Simple wrapper that returns just the content string (no tool support).
        """
        msg = self.chat(messages)
        return msg.get("content") or ""

    def describe_image(
        self,
        image_base64: str,
        prompt: str = "Describe this image in detail.",
        system: str = "",
        mime_type: str = "image/jpeg",
    ) -> str:
        """
        Send an image to the configured vision-capable backend.
        """
        backend = self._get_backend()
        if isinstance(backend, _ChatCompletionsBackend):
            return self._describe_chat_completions_images(
                backend, [image_base64], prompt, system, mime_type
            )

        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": prompt,
            "images": [image_base64],
        })

        model_name = backend._resolve_model_name() if hasattr(backend, "_resolve_model_name") else backend.model_name

        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
        }

        try:
            endpoint = f"{self.base_url}/api/chat"
            resp = requests.post(endpoint, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            msg = data.get("message") or {}
            return msg.get("content") or "(No description generated)"
        except Exception as exc:
            return f"[Vision error] {exc}"

    def describe_images(
        self,
        images_base64: List[str],
        prompt: str = "Describe these images in detail.",
        system: str = "",
        mime_type: str = "image/jpeg",
    ) -> str:
        """
        Send multiple images to the configured vision-capable backend.
        """
        backend = self._get_backend()
        if isinstance(backend, _ChatCompletionsBackend):
            return self._describe_chat_completions_images(
                backend, images_base64, prompt, system, mime_type
            )

        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": prompt,
            "images": images_base64,
        })

        model_name = backend._resolve_model_name() if hasattr(backend, "_resolve_model_name") else backend.model_name

        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
        }

        try:
            endpoint = f"{self.base_url}/api/chat"
            resp = requests.post(endpoint, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            msg = data.get("message") or {}
            return msg.get("content") or "(No description generated)"
        except Exception as exc:
            return f"[Vision error] {exc}"

    def _describe_chat_completions_images(
        self,
        backend: _ChatCompletionsBackend,
        images_base64: List[str],
        prompt: str,
        system: str,
        mime_type: str,
    ) -> str:
        """Call a chat-completions-compatible vision endpoint with data-URL images."""
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image}"},
            }
            for image in images_base64
        )
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        endpoint = backend._chat_endpoint()
        headers = {
            "Authorization": f"Bearer {backend.api_key}",
            "Content-Type": "application/json",
        }
        if not backend.api_key:
            return f"[Vision error] No API key found for {backend.label}."
        payload = {
            "model": backend.model_name,
            "messages": messages,
            "stream": False,
        }
        try:
            response = requests.post(
                endpoint, json=payload, headers=headers, timeout=backend.timeout
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                return "(No description generated)"
            result = choices[0].get("message", {}).get("content") or ""
            if isinstance(result, list):
                result = "\n".join(
                    part.get("text", "") for part in result if isinstance(part, dict)
                )
            return result or "(No description generated)"
        except Exception as exc:
            return f"[Vision error] {exc}"

