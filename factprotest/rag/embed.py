from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

import numpy as np

from factprotest.config import MODELS


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


class Embedder:
    """bge-small-en-v1.5 via ONNX (fastembed): 384-d, CPU-only, ~70 MB, no torch dependency."""

    def __init__(self, model_name: str = MODELS.embedding):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name)

    def documents(self, texts: Iterable[str], batch_size: int = 64) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return _normalize(np.array(list(self._model.passage_embed(texts, batch_size=batch_size)), dtype=np.float32))

    def queries(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return _normalize(np.array(list(self._model.query_embed(texts)), dtype=np.float32))


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()
