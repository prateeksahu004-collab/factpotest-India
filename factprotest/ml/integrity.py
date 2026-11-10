"""Coordinated inauthentic behaviour detection.

Flags groups of >= `min_size` posts on the same protest that are semantically near-identical
(cosine >= threshold) and posted within a short time window: the signature of copy-paste campaigns.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class CoordinatedCluster:
    member_ids: list[str]
    protest_id: int
    mean_similarity: float
    span_minutes: float
    distinct_authors: int
    dominant_stance: str

    @property
    def size(self) -> int:
        return len(self.member_ids)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def detect_coordination(records: list[dict[str, Any]], embeddings: np.ndarray, threshold: float = 0.88,
                        window_minutes: float | None = 180, min_size: int = 3) -> list[CoordinatedCluster]:
    """records need id, protest_id, created_at, author, stance; embeddings must be L2-normalised."""
    n = len(records)
    if n < min_size:
        return []
    sims = embeddings @ embeddings.T
    protest = np.array([str(r["protest_id"]) for r in records])
    times = np.array([_ts(r["created_at"]).timestamp() / 60 for r in records])
    candidate = (sims >= threshold) & (protest[:, None] == protest[None, :])
    if window_minutes is not None:
        candidate &= np.abs(times[:, None] - times[None, :]) <= window_minutes
    np.fill_diagonal(candidate, False)
    uf = _UnionFind(n)
    for i, j in zip(*np.nonzero(np.triu(candidate))):
        uf.union(int(i), int(j))
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    clusters = []
    for members in groups.values():
        if len(members) < min_size:
            continue
        sub = sims[np.ix_(members, members)]
        stances = [records[i]["stance"] for i in members]
        clusters.append(CoordinatedCluster(
            member_ids=[str(records[i]["id"]) for i in members], protest_id=int(records[members[0]]["protest_id"]),
            mean_similarity=round(float(sub[np.triu_indices(len(members), 1)].mean()), 4),
            span_minutes=round(float(times[members].max() - times[members].min()), 1),
            distinct_authors=len({records[i]["author"] for i in members}),
            dominant_stance=max(set(stances), key=stances.count),
        ))
    return sorted(clusters, key=lambda c: -c.size)


def post_level_scores(clusters: list[CoordinatedCluster], truth_ids: set[str]) -> dict[str, float]:
    flagged = {m for c in clusters for m in c.member_ids}
    tp = len(flagged & truth_ids)
    precision = tp / len(flagged) if flagged else 0.0
    recall = tp / len(truth_ids) if truth_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"flagged": len(flagged), "true_positives": tp, "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4)}


def duplication_rate(embeddings: np.ndarray, threshold: float = 0.95) -> float:
    """Share of posts that have at least one near-duplicate elsewhere in the set."""
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, 0)
    return round(float((sims >= threshold).any(axis=1).mean()), 4)
