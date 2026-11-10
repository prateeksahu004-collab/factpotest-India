"""Groq client wrapper: schema-validated JSON calls, tool calls, retries and per-model rate limiting."""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RETRY_WAIT = 120.0  # a daily-quota 429 can carry an hour-long retry-after; fail the batch instead of hanging


class LLMError(RuntimeError):
    pass


@dataclass
class CallRecord:
    model: str
    purpose: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ok: bool = True
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = _FENCE_RE.sub("", text.strip())
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = _OBJECT_RE.search(cleaned)
        if not match:
            raise
        value = json.loads(match.group())
    if not isinstance(value, dict):
        raise json.JSONDecodeError("expected a JSON object", cleaned, 0)
    return value


class TokenRateLimiter:
    """Sliding 60 s window per model so batch jobs stay under the provider's tokens-per-minute cap."""

    def __init__(self, tokens_per_minute: int = 7000):
        self.tpm = tokens_per_minute
        self._windows: dict[str, deque[tuple[float, int]]] = {}
        self._lock = threading.Lock()

    def _used(self, model: str, now: float) -> int:
        window = self._windows.setdefault(model, deque())
        while window and now - window[0][0] > 60:
            window.popleft()
        return sum(n for _, n in window)

    def wait(self, model: str, estimate: int) -> None:
        estimate = min(estimate, self.tpm)
        while True:
            with self._lock:
                now = time.monotonic()
                if self._used(model, now) + estimate <= self.tpm:
                    return
                oldest = self._windows[model][0][0]
            time.sleep(max(0.5, 60 - (now - oldest)))

    def record(self, model: str, tokens: int) -> None:
        with self._lock:
            self._windows.setdefault(model, deque()).append((time.monotonic(), tokens))


def _retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    value = headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


class LLMClient:
    def __init__(self, api_key: str | None = None, client: Any = None, max_retries: int = 5,
                 rate_limiter: TokenRateLimiter | None = None):
        if client is None:
            from groq import Groq

            client = Groq(api_key=api_key, max_retries=0, timeout=90)
        self._client = client
        self.max_retries = max_retries
        self.limiter = rate_limiter
        self.records: deque[CallRecord] = deque(maxlen=1000)
        self.retries = 0

    def _create(self, purpose: str, **kwargs: Any) -> Any:
        model = kwargs["model"]
        estimate = sum(len(str(m.get("content", ""))) for m in kwargs["messages"]) // 4 + kwargs.get("max_tokens", 1000) // 3
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            if self.limiter:
                self.limiter.wait(model, estimate)
            start = time.perf_counter()
            try:
                response = self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                latency = (time.perf_counter() - start) * 1000
                status = getattr(exc, "status_code", None)
                if self.limiter:
                    self.limiter.record(model, estimate // 2)
                if status in RETRYABLE_STATUS and attempt < self.max_retries:
                    wait = min(_retry_after(exc) or delay, MAX_RETRY_WAIT)
                    self.retries += 1
                    logger.warning("%s %s on %s; retrying in %.1fs: %s", purpose, status, model, wait, str(exc)[:300])
                    time.sleep(wait)
                    delay = min(delay * 2, 60)
                    continue
                self.records.append(CallRecord(model, purpose, latency, ok=False, error=str(exc)[:300]))
                raise LLMError(str(exc)) from exc
            latency = (time.perf_counter() - start) * 1000
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            if self.limiter:
                self.limiter.record(model, prompt_tokens + completion_tokens)
            self.records.append(CallRecord(model, purpose, latency, prompt_tokens, completion_tokens))
            return response
        raise LLMError("unreachable")

    @staticmethod
    def _reasoning_kwargs(model: str, effort: str) -> dict[str, Any]:
        return {"reasoning_effort": effort} if model.startswith("openai/gpt-oss") else {}

    def json(self, *, system: str, user: str, schema: type[T], model: str, purpose: str,
             temperature: float = 0.0, max_tokens: int = 2000, reasoning_effort: str = "low") -> T:
        """Call the model in JSON mode and validate against a Pydantic schema, repairing once on failure."""
        messages: list[dict[str, str]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_error: Exception | None = None
        for _ in range(3):
            try:
                response = self._create(
                    purpose, model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
                    response_format={"type": "json_object"}, **self._reasoning_kwargs(model, reasoning_effort),
                )
            except LLMError as exc:
                if "json_validate_failed" not in str(exc) and "max completion tokens" not in str(exc):
                    raise
                last_error = exc
                max_tokens = int(max_tokens * 1.5)
                continue
            content = response.choices[0].message.content or ""
            try:
                return schema.model_validate(parse_json_object(content))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                messages = messages + [
                    {"role": "assistant", "content": content[:4000]},
                    {"role": "user", "content": f"That JSON failed validation: {str(exc)[:600]}\nReturn corrected JSON only."},
                ]
        raise LLMError(f"{purpose}: no valid JSON after retries ({last_error})")

    def text(self, *, system: str, user: str, model: str, purpose: str, temperature: float = 0.3,
             max_tokens: int = 1500, reasoning_effort: str = "low") -> str:
        response = self._create(
            purpose, model=model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **self._reasoning_kwargs(model, reasoning_effort),
        )
        return (response.choices[0].message.content or "").strip()

    def tool_step(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]], model: str,
                  purpose: str, max_tokens: int = 2000, reasoning_effort: str = "medium") -> Any:
        response = self._create(
            purpose, model=model, messages=messages, tools=tools, tool_choice="auto", temperature=0.2,
            max_tokens=max_tokens, **self._reasoning_kwargs(model, reasoning_effort),
        )
        return response.choices[0].message

    def classify(self, *, model: str, text: str, purpose: str, system: str | None = None,
                 max_tokens: int = 50) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if system:
            kwargs.update(max_tokens=max_tokens, **self._reasoning_kwargs(model, "low"))
        response = self._create(purpose, **kwargs)
        return (response.choices[0].message.content or "").strip()

    def usage_summary(self) -> dict[str, Any]:
        records = list(self.records)
        ok = [r for r in records if r.ok]
        latencies = sorted(r.latency_ms for r in ok)
        return {
            "calls": len(records),
            "failed": len(records) - len(ok),
            "prompt_tokens": sum(r.prompt_tokens for r in ok),
            "completion_tokens": sum(r.completion_tokens for r in ok),
            "p50_latency_ms": round(latencies[len(latencies) // 2], 1) if latencies else None,
            "p95_latency_ms": round(latencies[int(len(latencies) * 0.95) - 1], 1) if latencies else None,
        }
