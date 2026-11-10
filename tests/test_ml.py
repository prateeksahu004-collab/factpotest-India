from datetime import datetime, timedelta

import numpy as np

from factprotest.ml.integrity import detect_coordination, post_level_scores
from factprotest.ml.stance import LABELS, grouped_cv, tfidf_logreg
from factprotest.ml.topics import class_tfidf_keywords, discover_topics


def _records(n: int, minutes_apart: float, protest_id: int = 1) -> list[dict]:
    start = datetime(2026, 11, 1)
    return [{"id": f"r{protest_id}-{i}", "protest_id": protest_id, "author": f"a{i}", "stance": "Oppose",
             "created_at": (start + timedelta(minutes=i * minutes_apart)).isoformat()} for i in range(n)]


def test_coordination_needs_similarity_and_time_proximity():
    base = np.ones(8, dtype=np.float32) / np.sqrt(8)
    vectors = np.stack([base] * 4)
    burst = detect_coordination(_records(4, 10), vectors, threshold=0.9, window_minutes=60)
    assert len(burst) == 1 and burst[0].size == 4
    spread = detect_coordination(_records(4, 600), vectors, threshold=0.9, window_minutes=60)
    assert spread == []
    assert post_level_scores(burst, {"r1-0", "r1-1", "r1-2", "r1-3"})["f1"] == 1.0


def test_coordination_ignores_cross_protest_pairs():
    base = np.ones(8, dtype=np.float32) / np.sqrt(8)
    records = _records(2, 1, protest_id=1) + _records(2, 1, protest_id=2)
    assert detect_coordination(records, np.stack([base] * 4), threshold=0.9, min_size=3) == []


def test_grouped_cv_holds_out_whole_protests():
    support = ["farmers deserve msp support the protest", "stand with the protesters fully", "i back this movement strongly"]
    oppose = ["this protest is wrong and disruptive", "oppose the agitation it hurts people", "against these protesters"]
    neutral = ["both sides have valid points", "wait for dialogue and data", "mixed feelings on this issue"]
    texts, y, groups = [], [], []
    for g in range(3):
        for label, pool in zip(LABELS, (oppose, neutral, support)):
            for t in pool:
                texts.append(f"{t} {g}")
                y.append(label)
                groups.append(g)
    report, preds = grouped_cv(tfidf_logreg(), texts, np.array(y), np.array(groups))
    assert report["n"] == len(texts) and len(preds) == len(texts)
    assert report["macro_f1"] > 0.5


def test_topics_split_distinct_themes():
    texts = ["msp guarantee for farmers"] * 5 + ["paper leak exam cancelled"] * 5
    vectors = np.array([[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 5) + np.random.default_rng(0).normal(0, 0.01, (10, 2))
    topics, silhouettes = discover_topics(texts, vectors, k_range=range(2, 4))
    assert len(topics) == 2 and silhouettes[2] > 0.9
    keywords = class_tfidf_keywords(texts, np.array([0] * 5 + [1] * 5))
    assert "msp" in keywords[0] and "leak" in keywords[1]
