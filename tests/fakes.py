"""Offline stand-ins for the Groq client and the embedding model."""
from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np


def completion(content: str = "", tool_calls: list[Any] | None = None) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


class FakeGroq:
    """Mimics `client.chat.completions.create`; `responder(kwargs)` returns a completion or raises."""

    def __init__(self, responder: Callable[[dict[str, Any]], Any]):
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._responder = responder

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._responder(kwargs)


class HashEncoder:
    """Deterministic bag-of-words embedding so dense retrieval can be tested without downloading a model."""

    dim = 64

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            v[hash(token.strip(".,")) % self.dim] += 1
        return v / max(np.linalg.norm(v), 1e-9)

    def documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def queries(self, texts: list[str]) -> np.ndarray:
        return self.documents(texts)
