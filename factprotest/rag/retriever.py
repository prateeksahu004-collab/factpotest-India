"""Hybrid retrieval: BM25 (lexical) + dense bge embeddings, fused with Reciprocal Rank Fusion."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from factprotest.rag.kb import Chunk

Mode = Literal["bm25", "dense", "hybrid"]
TOKEN_RE = re.compile(r"[a-z0-9]+")


class QueryEncoder(Protocol):
    def queries(self, texts: list[str]) -> np.ndarray: ...


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25_rank: int | None
    dense_rank: int | None


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, encoder: QueryEncoder | None, rrf_k: int = 60):
        self.chunks = chunks
        self.embeddings = embeddings
        self.encoder = encoder
        self.rrf_k = rrf_k
        self._bm25 = BM25Okapi([tokenize(c.text + " " + c.section) for c in chunks])
        self._protest_ids = np.array([c.protest_id for c in chunks])

    def _mask(self, protest_id: int | None) -> np.ndarray:
        if protest_id is None:
            return np.ones(len(self.chunks), dtype=bool)
        mask = self._protest_ids == protest_id
        return mask if mask.any() else np.ones(len(self.chunks), dtype=bool)

    def _bm25_ranking(self, query: str, mask: np.ndarray, depth: int) -> list[int]:
        scores = np.where(mask, self._bm25.get_scores(tokenize(query)), -np.inf)
        order = np.argsort(-scores)[:depth]
        return [int(i) for i in order if np.isfinite(scores[i]) and scores[i] > 0]

    def _dense_ranking(self, query_vec: np.ndarray, mask: np.ndarray, depth: int) -> tuple[list[int], np.ndarray]:
        sims = np.where(mask, self.embeddings @ query_vec, -np.inf)
        return [int(i) for i in np.argsort(-sims)[:depth] if np.isfinite(sims[i])], sims

    def search(self, query: str, k: int = 4, protest_id: int | None = None, mode: Mode = "hybrid",
               depth: int = 40, query_vec: np.ndarray | None = None) -> list[Hit]:
        mask = self._mask(protest_id)
        bm25 = self._bm25_ranking(query, mask, depth) if mode in ("bm25", "hybrid") else []
        dense: list[int] = []
        if mode in ("dense", "hybrid"):
            if query_vec is None:
                if self.encoder is None:
                    raise ValueError("dense retrieval needs a query encoder or a precomputed query vector")
                query_vec = self.encoder.queries([query])[0]
            dense, _ = self._dense_ranking(query_vec, mask, depth)
        if mode == "bm25":
            fused = {idx: 1.0 / (self.rrf_k + r + 1) for r, idx in enumerate(bm25)}
        elif mode == "dense":
            fused = {idx: 1.0 / (self.rrf_k + r + 1) for r, idx in enumerate(dense)}
        else:
            fused = reciprocal_rank_fusion([bm25, dense], self.rrf_k)
        bm25_pos = {idx: r for r, idx in enumerate(bm25)}
        dense_pos = {idx: r for r, idx in enumerate(dense)}
        top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [Hit(self.chunks[i], s, bm25_pos.get(i), dense_pos.get(i)) for i, s in top]

    def search_many(self, queries: list[str], k: int = 4, protest_ids: list[int | None] | None = None,
                    mode: Mode = "hybrid") -> list[list[Hit]]:
        protest_ids = protest_ids or [None] * len(queries)
        vecs = self.encoder.queries(queries) if (mode != "bm25" and self.encoder is not None and queries) else None
        return [
            self.search(q, k=k, protest_id=pid, mode=mode, query_vec=None if vecs is None else vecs[i])
            for i, (q, pid) in enumerate(zip(queries, protest_ids))
        ]
