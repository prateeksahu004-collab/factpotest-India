"""Topic discovery: KMeans over sentence embeddings (k picked by silhouette) + class-based TF-IDF keywords."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics import silhouette_score

EXTRA_STOP = {"protest", "protests", "people", "government", "india", "indian", "just", "like", "really", "think",
              "hai", "ki", "ka", "ke", "ko", "se", "bhi", "nahi", "toh", "aur", "yeh", "ye", "hain", "ho", "kya", "na"}


@dataclass
class Topic:
    cluster: int
    size: int
    keywords: list[str]
    representative_idx: list[int]
    label: str = ""
    summary: str = ""
    stance_mix: dict[str, int] = field(default_factory=dict)


def choose_k(embeddings: np.ndarray, k_range: range, seed: int = 0) -> tuple[int, dict[int, float]]:
    scores = {}
    for k in k_range:
        if k >= len(embeddings):
            break
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(embeddings)
        scores[k] = round(float(silhouette_score(embeddings, labels, metric="cosine")), 4)
    best = max(scores, key=scores.get) if scores else 1
    return best, scores


def class_tfidf_keywords(texts: list[str], labels: np.ndarray, top_n: int = 8) -> dict[int, list[str]]:
    """c-TF-IDF (as in BERTopic): treat each cluster as one document, weight terms by cluster frequency
    times log(1 + average words per cluster / term frequency across clusters)."""
    clusters = sorted(set(labels.tolist()))
    docs = [" ".join(t for t, lab in zip(texts, labels) if lab == c) for c in clusters]
    vectorizer = CountVectorizer(stop_words=list(ENGLISH_STOP_WORDS | EXTRA_STOP), ngram_range=(1, 2),
                                 min_df=1, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b")
    counts = vectorizer.fit_transform(docs).toarray().astype(float)
    tf = counts / np.clip(counts.sum(axis=1, keepdims=True), 1, None)
    avg_words = counts.sum() / len(clusters)
    idf = np.log1p(avg_words / np.clip(counts.sum(axis=0), 1, None))
    weights = tf * idf
    vocab = np.array(vectorizer.get_feature_names_out())
    return {c: vocab[np.argsort(-weights[i])[:top_n]].tolist() for i, c in enumerate(clusters)}


def discover_topics(texts: list[str], embeddings: np.ndarray, stances: list[str] | None = None,
                    k_range: range = range(2, 7), seed: int = 0) -> tuple[list[Topic], dict[int, float]]:
    if len(texts) < 6:
        return [], {}
    k, silhouettes = choose_k(embeddings, k_range, seed)
    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(embeddings)
    keywords = class_tfidf_keywords(texts, model.labels_)
    topics = []
    for c in range(k):
        idx = np.where(model.labels_ == c)[0]
        dist = np.linalg.norm(embeddings[idx] - model.cluster_centers_[c], axis=1)
        mix: dict[str, int] = {}
        for i in idx:
            if stances:
                mix[stances[i]] = mix.get(stances[i], 0) + 1
        topics.append(Topic(c, len(idx), keywords[c], idx[np.argsort(dist)[:3]].tolist(), stance_mix=mix))
    return sorted(topics, key=lambda t: -t.size), silhouettes
