"""Stance classification (Support / Oppose / Neutral) with leave-one-protest-out evaluation.

Grouped CV matters here: a random split lets the model memorise protest-specific vocabulary
("MSP", "NTA", "Hindenburg") and overstates how well it handles a protest it has never seen.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import FeatureUnion, Pipeline

LABELS = ["Oppose", "Neutral", "Support"]


def tfidf_logreg() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=40000)),
    ])
    return Pipeline([("features", features),
                     ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=4.0))])


def embedding_logreg() -> LogisticRegression:
    return LogisticRegression(max_iter=3000, class_weight="balanced", C=2.0)


def summarize(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    return {
        "n": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0), 4),
        "per_class": {k: {m: round(v, 4) for m, v in vals.items()}
                      for k, vals in classification_report(y_true, y_pred, labels=LABELS, output_dict=True,
                                                           zero_division=0).items() if k in LABELS},
        "confusion_matrix": {"labels": LABELS, "matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist()},
    }


def grouped_cv(model: Any, X: Any, y: np.ndarray, groups: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    """Leave-one-protest-out: train on 9 protests, test on the held-out one, repeat for each."""
    predictions = np.empty(len(y), dtype=object)
    fold_f1 = []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        fitted = clone(model).fit(X[train_idx] if not isinstance(X, list) else [X[i] for i in train_idx], y[train_idx])
        test_X = X[test_idx] if not isinstance(X, list) else [X[i] for i in test_idx]
        predictions[test_idx] = fitted.predict(test_X)
        fold_f1.append(f1_score(y[test_idx], predictions[test_idx], labels=LABELS, average="macro", zero_division=0))
    report = summarize(list(y), list(predictions))
    report["fold_macro_f1_mean"] = round(float(np.mean(fold_f1)), 4)
    report["fold_macro_f1_std"] = round(float(np.std(fold_f1)), 4)
    return report, predictions


@dataclass
class StanceModel:
    pipeline: Any
    kind: str
    metrics: dict[str, Any]

    def predict(self, texts: list[str], embed: Callable[[list[str]], np.ndarray] | None = None) -> list[str]:
        X = embed(texts) if self.kind == "embedding" and embed else texts
        return list(self.pipeline.predict(X))

    def predict_proba(self, texts: list[str], embed: Callable[[list[str]], np.ndarray] | None = None) -> list[dict[str, float]]:
        X = embed(texts) if self.kind == "embedding" and embed else texts
        probs = self.pipeline.predict_proba(X)
        classes = list(self.pipeline.classes_)
        return [{c: round(float(p[classes.index(c)]), 4) for c in LABELS} for p in probs]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "kind": self.kind, "metrics": self.metrics}, path)

    @classmethod
    def load(cls, path: Path) -> StanceModel:
        blob = joblib.load(path)
        return cls(blob["pipeline"], blob["kind"], blob["metrics"])
