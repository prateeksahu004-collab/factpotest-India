"""Train and evaluate the classical-ML components on the scored corpus and write artifacts/.

  stance     TF-IDF+LogReg vs bge-embedding+LogReg vs LLM zero-shot, leave-one-protest-out CV
  topics     per-protest KMeans topics (k by silhouette) + c-TF-IDF keywords + LLM-written labels
  integrity  coordinated-campaign detector vs injected ground truth, threshold sweep, legacy-data audit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factprotest import prompts  # noqa: E402
from factprotest.config import ARTIFACTS_DIR, CORPUS_DIR, MODELS, secret  # noqa: E402
from factprotest.llm import LLMClient, LLMError, TokenRateLimiter  # noqa: E402
from factprotest.ml import integrity  # noqa: E402
from factprotest.ml.stance import StanceModel, embedding_logreg, grouped_cv, summarize, tfidf_logreg  # noqa: E402
from factprotest.ml.topics import discover_topics  # noqa: E402
from factprotest.rag.embed import get_embedder  # noqa: E402

SCORED = CORPUS_DIR / "opinions_scored.jsonl"
METRICS = ARTIFACTS_DIR / "metrics"


class TopicLabel(BaseModel):
    cluster: int
    label: str
    summary: str


class TopicLabels(BaseModel):
    topics: list[TopicLabel]


def load_corpus() -> list[dict[str, Any]]:
    """Full raw corpus, with pipeline assessments attached where scoring has run (None otherwise)."""
    scored = {r["id"]: r["assessment"] for r in map(json.loads, SCORED.open())} if SCORED.exists() else {}
    return [{**r, "assessment": scored.get(r["id"])} for r in map(json.loads, (CORPUS_DIR / "opinions.jsonl").open())]


def corpus_embeddings(records: list[dict[str, Any]]) -> np.ndarray:
    path, ids_path = ARTIFACTS_DIR / "opinion_embeddings.npy", ARTIFACTS_DIR / "opinion_embedding_ids.json"
    ids = [r["id"] for r in records]
    if path.exists() and ids_path.exists() and json.loads(ids_path.read_text()) == ids:
        return np.load(path)
    vectors = get_embedder().documents([r["text"] for r in records])
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    ids_path.write_text(json.dumps(ids))
    return vectors


def dump(name: str, payload: dict[str, Any]) -> None:
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / f"{name}.json").write_text(json.dumps(
        {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **payload}, indent=2, ensure_ascii=False, default=str))


def train_stance(records: list[dict[str, Any]], vectors: np.ndarray) -> None:
    organic = [i for i, r in enumerate(records) if not r.get("campaign_id")]
    texts = [records[i]["text"] for i in organic]
    y = np.array([records[i]["stance"] for i in organic])
    groups = np.array([records[i]["protest_id"] for i in organic])
    tfidf_report, tfidf_pred = grouped_cv(tfidf_logreg(), texts, y, groups)
    emb_report, emb_pred = grouped_cv(embedding_logreg(), vectors[organic], y, groups)
    best_kind = "embedding" if emb_report["macro_f1"] >= tfidf_report["macro_f1"] else "tfidf"
    model = (embedding_logreg().fit(vectors[organic], y) if best_kind == "embedding" else tfidf_logreg().fit(texts, y))
    StanceModel(model, best_kind, {"tfidf_logreg": tfidf_report, "embedding_logreg": emb_report}).save(
        ARTIFACTS_DIR / "stance_model.joblib")

    # Head-to-head with the LLM only where the pipeline has scored the opinion, using out-of-fold ML predictions.
    both = [j for j, i in enumerate(organic) if records[i]["assessment"]]
    head_to_head = {}
    if both:
        truth = [y[j] for j in both]
        head_to_head = {
            "n": len(both),
            "tfidf_word+char_logreg": summarize(truth, [tfidf_pred[j] for j in both]),
            "bge_small_embedding_logreg": summarize(truth, [emb_pred[j] for j in both]),
            f"llm_zero_shot ({MODELS.analyze})": summarize(truth, [records[organic[j]]["assessment"]["predicted_stance"] for j in both]),
        }
    dump("stance", {
        "protocol": "leave-one-protest-out cross-validation (10 folds, train on 9 protests, test on the unseen one)",
        "n": len(organic), "label_distribution": {s: int((y == s).sum()) for s in sorted(set(y))},
        "deployed_model": best_kind,
        "models": {"tfidf_word+char_logreg": tfidf_report, "bge_small_embedding_logreg": emb_report},
        "head_to_head_vs_llm_on_scored_subset": head_to_head,
    })
    llm_f1 = next((v["macro_f1"] for k, v in head_to_head.items() if k.startswith("llm")), None)
    print(f"stance macro-F1  tfidf={tfidf_report['macro_f1']}  embedding={emb_report['macro_f1']}  llm(subset)={llm_f1}")


def train_topics(records: list[dict[str, Any]], vectors: np.ndarray, label_with_llm: bool) -> None:
    llm = LLMClient(api_key=secret("GROQ_API_KEY"), rate_limiter=TokenRateLimiter(7000)) if label_with_llm else None
    output: dict[str, Any] = {}
    for protest_id in sorted({r["protest_id"] for r in records}):
        idx = [i for i, r in enumerate(records) if r["protest_id"] == protest_id and not r.get("campaign_id")]
        texts = [records[i]["text"] for i in idx]
        topics, silhouettes = discover_topics(texts, vectors[idx], [records[i]["stance"] for i in idx])
        if llm and topics:
            block = "\n\n".join(f"Cluster {t.cluster}: keywords={', '.join(t.keywords)}\n" + "\n".join(
                f"- {texts[j][:220]}" for j in t.representative_idx) for t in topics)
            try:
                labels = llm.json(system=prompts.TOPIC_SYSTEM, user=f"Protest: {records[idx[0]]['protest_title']}\n\n{block}",
                                  schema=TopicLabels, model=MODELS.analyze, purpose="topics", max_tokens=1500)
                by_cluster = {lab.cluster: lab for lab in labels.topics}
                for t in topics:
                    if t.cluster in by_cluster:
                        t.label, t.summary = by_cluster[t.cluster].label, by_cluster[t.cluster].summary
            except LLMError as exc:
                print(f"  topic labelling failed for P{protest_id}: {exc}")
        output[str(protest_id)] = [{
            "cluster": t.cluster, "label": t.label or ", ".join(t.keywords[:3]), "summary": t.summary, "size": t.size,
            "keywords": t.keywords, "stance_mix": t.stance_mix,
            "examples": [{"ref": f"O:{records[idx[j]]['id'][:8]}", "text": texts[j][:240]} for j in t.representative_idx],
        } for t in topics]
        output[f"{protest_id}_silhouette"] = silhouettes
        print(f"P{protest_id}: k={len(topics)} silhouettes={silhouettes}")
    (ARTIFACTS_DIR / "topics.json").write_text(json.dumps(output, indent=2, ensure_ascii=False))


def train_integrity(records: list[dict[str, Any]], vectors: np.ndarray, legacy: bool) -> None:
    truth = {r["id"] for r in records if r.get("campaign_id")}
    sweep = {}
    for threshold in (0.80, 0.84, 0.88, 0.92, 0.95):
        clusters = integrity.detect_coordination(records, vectors, threshold=threshold)
        sweep[str(threshold)] = {**integrity.post_level_scores(clusters, truth), "clusters": len(clusters)}
    chosen = integrity.detect_coordination(records, vectors, threshold=0.88)
    no_time = integrity.detect_coordination(records, vectors, threshold=0.88, window_minutes=None)
    payload: dict[str, Any] = {
        "ground_truth_campaign_posts": len(truth), "deployed_threshold": 0.88, "time_window_minutes": 180,
        "threshold_sweep": sweep,
        "ablation_without_time_window": integrity.post_level_scores(no_time, truth),
        "clusters": [c.__dict__ for c in chosen],
        "organic_duplication_rate@0.95": integrity.duplication_rate(
            vectors[[i for i, r in enumerate(records) if not r.get("campaign_id")]]),
    }
    if legacy:
        from factprotest.data import SupabaseRepository

        rows = [o for o in SupabaseRepository().opinions() if '"schema_version": 2' not in (o.get("ai_analysis") or "")]
        legacy_vectors = get_embedder().documents([o["opinion_text"] for o in rows])
        payload["legacy_v1_seed_audit"] = {
            "opinions": len(rows),
            "duplication_rate@0.95": integrity.duplication_rate(legacy_vectors, 0.95),
            "unique_texts_exact": len({o["opinion_text"] for o in rows}),
            "note": "v1 demo data was template-generated with random scores; this audit is why it was replaced",
        }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "coordination.json").write_text(json.dumps([c.__dict__ for c in chosen], indent=2))
    dump("integrity", payload)
    print(f"integrity @0.88: {sweep['0.88']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parts", nargs="*", default=["stance", "topics", "integrity"])
    parser.add_argument("--no-llm-labels", action="store_true")
    parser.add_argument("--legacy-audit", action="store_true")
    args = parser.parse_args()
    records = load_corpus()
    vectors = corpus_embeddings(records)
    print(f"{len(records)} scored opinions, embeddings {vectors.shape}")
    if "stance" in args.parts:
        train_stance(records, vectors)
    if "topics" in args.parts:
        train_topics(records, vectors, not args.no_llm_labels)
    if "integrity" in args.parts:
        train_integrity(records, vectors, args.legacy_audit)


if __name__ == "__main__":
    main()
