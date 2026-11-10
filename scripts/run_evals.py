"""Evaluation harness. Each suite writes evals/results/<suite>.json; `summary` merges them.

  retrieval    Recall@k / MRR of BM25 vs dense vs hybrid on gold claims          (no LLM calls)
  verify       claim-verification accuracy / macro-F1 for several configurations (LLM)
  consistency  run-to-run stability of the credibility score                      (LLM)
  endtoend     misinformation recall, false-alarm rate, stance and fairness on the scored corpus (no LLM)
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factprotest import analytics, prompts  # noqa: E402
from factprotest.config import CORPUS_DIR, EVAL_DIR, KB_DIR, MODELS, secret  # noqa: E402
from factprotest.factcheck import ClaimTask, OpinionInput, assess, verify  # noqa: E402
from factprotest.llm import LLMClient, LLMError, TokenRateLimiter  # noqa: E402
from factprotest.ml.stance import summarize  # noqa: E402
from factprotest.rag.embed import get_embedder  # noqa: E402
from factprotest.rag.kb import load_kb  # noqa: E402
from factprotest.rag.retriever import Hit, HybridRetriever  # noqa: E402
from factprotest.schemas import VERDICTS, BatchStance, ClaimVerdict  # noqa: E402

RESULTS = EVAL_DIR / "results"
GOLD = EVAL_DIR / "gold_claims.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write(name: str, payload: dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {"suite": name, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **payload}
    (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    print(json.dumps({k: v for k, v in payload.items() if k not in ("predictions", "per_claim")}, indent=2, default=str)[:3000])


def retriever() -> HybridRetriever:
    chunks, embeddings = load_kb(KB_DIR)
    return HybridRetriever(chunks, embeddings, get_embedder())


def _neighbours(chunk_id: str) -> set[str]:
    stem, _, idx = chunk_id.rpartition("-")
    return {f"{stem}-{int(idx) + d:03d}" for d in (-1, 0, 1)}


class Paraphrases(BaseModel):
    paraphrases: list[dict[str, str]]


def paraphrased_gold(gold: list[dict[str, Any]]) -> dict[str, str]:
    """Heavy paraphrases of gold claims (different vocabulary, same meaning), cached on disk."""
    path = EVAL_DIR / "gold_claims_paraphrased.json"
    cached: dict[str, str] = json.loads(path.read_text()) if path.exists() else {}
    todo = [g for g in gold if g["id"] not in cached]
    llm = LLMClient(api_key=secret("GROQ_API_KEY"), rate_limiter=TokenRateLimiter(7000)) if todo else None
    for start in range(0, len(todo), 12):
        batch = todo[start:start + 12]
        out = llm.json(  # type: ignore[union-attr]
            system="Rewrite each claim the way an ordinary citizen might say it online: keep the meaning and every "
                   "number, date and name, but change as many other words and the sentence structure as possible. "
                   'Return JSON {"paraphrases": [{"id": "...", "text": "..."}]}',
            user="\n".join(f'{g["id"]}: {g["claim"]}' for g in batch), schema=Paraphrases,
            model=MODELS.gold_generator, purpose="paraphrase", temperature=0.7, max_tokens=900)
        cached.update({p["id"]: p["text"] for p in out.paraphrases if "id" in p and "text" in p})
        path.write_text(json.dumps(cached, indent=2, ensure_ascii=False))
    return cached


def suite_retrieval(paraphrase: bool = False) -> None:
    gold = [g for g in load_jsonl(GOLD) if g["label"] != "NOT_ENOUGH_INFO"]
    if paraphrase:
        rewrites = paraphrased_gold(gold)
        gold = [{**g, "claim": rewrites[g["id"]]} for g in gold if g["id"] in rewrites]
    r = retriever()
    query_vecs = r.encoder.queries([g["claim"] for g in gold])
    results: dict[str, Any] = {}
    for mode in ("bm25", "dense", "hybrid"):
        strict, relaxed = {1: 0, 3: 0, 5: 0}, {1: 0, 3: 0, 5: 0}
        rr = []
        for g, vec in zip(gold, query_vecs):
            ids = [h.chunk.chunk_id for h in r.search(g["claim"], k=10, protest_id=g["protest_id"], mode=mode, query_vec=vec)]
            near = _neighbours(g["source_chunk_id"])
            for k in strict:
                strict[k] += g["source_chunk_id"] in ids[:k]
                relaxed[k] += bool(near & set(ids[:k]))
            rr.append(1 / (ids.index(g["source_chunk_id"]) + 1) if g["source_chunk_id"] in ids else 0.0)
        results[mode] = {**{f"recall@{k}": round(v / len(gold), 4) for k, v in strict.items()},
                         **{f"relaxed_recall@{k}": round(v / len(gold), 4) for k, v in relaxed.items()},
                         "mrr@10": round(float(np.mean(rr)), 4)}
    note = "relaxed counts the gold passage's overlapping neighbours (+/-1 chunk) as hits"
    if paraphrase:
        note += "; queries are heavy paraphrases of the gold claims (different vocabulary, same facts)"
    write("retrieval_paraphrase" if paraphrase else "retrieval",
          {"n_queries": len(gold), "kb_chunks": len(r.chunks), "note": note, "results": results})


def _verdict_metrics(labels: list[str], preds: list[str]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    return {
        "accuracy": round(accuracy_score(labels, preds), 4),
        "macro_f1": round(f1_score(labels, preds, labels=list(VERDICTS), average="macro", zero_division=0), 4),
        "per_class": {k: {m: round(v, 4) for m, v in vals.items()} for k, vals in classification_report(
            labels, preds, labels=list(VERDICTS), output_dict=True, zero_division=0).items() if k in VERDICTS},
        "confusion_matrix": {"labels": list(VERDICTS), "matrix": confusion_matrix(labels, preds, labels=list(VERDICTS)).tolist()},
    }


def _cached_verify(llm: LLMClient, name: str, tasks: list[ClaimTask], model: str) -> tuple[dict[str, ClaimVerdict], int]:
    """Verify in batches, checkpointing each batch so a daily-quota stop loses nothing and a re-run resumes."""
    path = RESULTS / "verify_cache" / f"{name}.json"
    done: dict[str, dict[str, Any]] = json.loads(path.read_text()) if path.exists() else {}
    todo = [t for t in tasks if t.claim_id not in done]
    newly = 0
    for start in range(0, len(todo), 8):
        batch = todo[start:start + 8]
        try:
            result = verify(llm, batch, model=model)
        except LLMError as exc:
            print(f"{name}: stopped after {len(done)}/{len(tasks)} claims: {str(exc)[:160]}")
            break
        done.update({k: v.model_dump() for k, v in result.items()})
        for task in batch:
            done.setdefault(task.claim_id, {"claim_id": task.claim_id, "verdict": "NOT_ENOUGH_INFO",
                                            "confidence": 0.0, "explanation": "verifier returned no verdict"})
        newly += len(batch)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(done, ensure_ascii=False))
    return {k: ClaimVerdict(**v) for k, v in done.items()}, newly


def suite_verify(configs: list[str]) -> None:
    gold = load_jsonl(GOLD)
    r = retriever()
    by_id = {c.chunk_id: c for c in r.chunks}
    llm = LLMClient(api_key=secret("GROQ_API_KEY"), rate_limiter=TokenRateLimiter(7000))
    hybrid_hits = r.search_many([g["claim"] for g in gold], k=3, protest_ids=[g["protest_id"] for g in gold])
    bm25_hits = r.search_many([g["claim"] for g in gold], k=3, protest_ids=[g["protest_id"] for g in gold], mode="bm25")
    available = {
        "hybrid_k3__gpt-oss-120b": (hybrid_hits, MODELS.verify),
        "hybrid_k3__gpt-oss-20b": (hybrid_hits, MODELS.analyze),
        "bm25_k3__gpt-oss-120b": (bm25_hits, MODELS.verify),
        "oracle__gpt-oss-120b": ([[Hit(by_id[g["source_chunk_id"]], 1.0, None, None)] for g in gold], MODELS.verify),
    }
    out = json.loads((RESULTS / "verification.json").read_text()) if (RESULTS / "verification.json").exists() else {"configs": {}}
    for name in configs:
        hits, model = available[name]
        tasks = [ClaimTask(g["id"], g["claim"], h) for g, h in zip(gold, hits)]
        first = len(llm.records)
        start = time.perf_counter()
        verdicts, newly_done = _cached_verify(llm, name, tasks, model)
        seconds = time.perf_counter() - start
        calls = list(llm.records)[first:]
        if len(verdicts) < len(gold):
            print(f"{name}: {len(verdicts)}/{len(gold)} claims verified so far (quota); re-run to finish")
            continue
        preds = [verdicts[g["id"]].verdict for g in gold]
        labels = [g["label"] for g in gold]
        previous = out["configs"].get(name, {})
        out["configs"][name] = {
            **_verdict_metrics(labels, preds), "n": len(gold), "model": model, "missing": 0,
            "tokens_per_claim": round(sum(c.total_tokens for c in calls) / newly_done, 1) if newly_done
            else previous.get("tokens_per_claim"),
            "seconds_per_claim_incl_rate_limit": round(seconds / newly_done, 2) if newly_done
            else previous.get("seconds_per_claim_incl_rate_limit"),
            "per_claim": [{"id": g["id"], "label": g["label"], "pred": p,
                           "confidence": verdicts[g["id"]].confidence if g["id"] in verdicts else None}
                          for g, p in zip(gold, preds)],
        }
        print(f"{name}: acc={out['configs'][name]['accuracy']} macroF1={out['configs'][name]['macro_f1']}")
    out["prompt_version"] = prompts.VERIFY_VERSION
    out.pop("suite", None)
    out.pop("generated_at", None)
    write("verification", out)


def suite_consistency(n: int = 15, runs: int = 3) -> None:
    scored = load_jsonl(CORPUS_DIR / "opinions_scored.jsonl")
    sample = random.Random(3).sample(scored, n)
    r = retriever()
    llm = LLMClient(api_key=secret("GROQ_API_KEY"), rate_limiter=TokenRateLimiter(7000))
    scores: dict[str, list[int]] = {s["id"]: [s["assessment"]["score"]["final"]] for s in sample}
    stances: dict[str, list[str]] = {s["id"]: [s["assessment"]["predicted_stance"]] for s in sample}
    for _ in range(runs - 1):
        items = [OpinionInput(s["id"], s["protest_id"], s["protest_title"], s["stance"], s["text"]) for s in sample]
        for oid, a in assess(llm, r, items).items():
            scores[oid].append(a.score.final)
            stances[oid].append(a.predicted_stance)
    stdevs = [statistics.pstdev(v) for v in scores.values() if len(v) == runs]
    ranges = [max(v) - min(v) for v in scores.values() if len(v) == runs]
    write("consistency", {
        "opinions": n, "runs_per_opinion": runs,
        "mean_score_std": round(float(np.mean(stdevs)), 2), "median_score_range": float(np.median(ranges)),
        "max_score_range": int(max(ranges)), "share_within_10_points": round(float(np.mean([x <= 10 for x in ranges])), 3),
        "stance_agreement_all_runs": round(float(np.mean([len(set(v)) == 1 for v in stances.values()])), 3),
        "scores": scores,
    })


def wilson(hits: list[bool], z: float = 1.96) -> list[float]:
    """Wilson score interval for a proportion; well-behaved for small n and rates near 0 or 1."""
    n = len(hits)
    if n == 0:
        return [float("nan"), float("nan")]
    p = sum(hits) / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(float(centre - half), 4), round(float(centre + half), 4)]


def suite_endtoend() -> None:
    scored = load_jsonl(CORPUS_DIR / "opinions_scored.jsonl")
    organic = [s for s in scored if not s.get("campaign_id")]

    def refuted(s: dict[str, Any], threshold: float = 0.6) -> bool:
        return any(c["verdict"] == "REFUTED" and c["confidence"] >= threshold for c in s["assessment"]["claims"])

    false_fact = [s for s in organic if s.get("false_claim")]
    accurate = [s for s in organic if s["factual_mode"] == "accurate_fact"]
    no_fact = [s for s in organic if s["factual_mode"] == "no_fact"]
    scores_false = [s["assessment"]["score"]["final"] for s in false_fact]
    scores_accurate = [s["assessment"]["score"]["final"] for s in accurate]
    auc = roc_auc_score([0] * len(scores_false) + [1] * len(scores_accurate), scores_false + scores_accurate)
    labels = [s["stance"] for s in organic]
    llm_stance = [s["assessment"]["predicted_stance"] for s in organic]
    write("endtoend", {
        "n_scored": len(scored),
        "protests_covered": sorted({s["protest_id"] for s in scored}),
        "misinformation": {
            "opinions_with_injected_false_claim": len(false_fact),
            "opinions_accurate_fact": len(accurate),
            "opinions_no_fact": len(no_fact),
            "recall_refuted_flag": round(np.mean([refuted(s) for s in false_fact]), 4),
            "recall_wilson_95ci": wilson([refuted(s) for s in false_fact]),
            "false_alarm_rate_accurate_fact": round(np.mean([refuted(s) for s in accurate]), 4),
            "false_alarm_wilson_95ci": wilson([refuted(s) for s in accurate]),
            "false_alarm_rate_no_fact": round(np.mean([refuted(s) for s in no_fact]), 4),
            "auc_score_accurate_vs_false": round(float(auc), 4),
            "mean_score": {"false_fact": round(float(np.mean(scores_false)), 1),
                           "accurate_fact": round(float(np.mean(scores_accurate)), 1),
                           "no_fact": round(float(np.mean([s["assessment"]["score"]["final"] for s in no_fact])), 1)},
        },
        "claims_extracted_per_opinion": round(float(np.mean([len(s["assessment"]["claims"]) for s in organic])), 2),
        "verdict_distribution": {v: sum(c["verdict"] == v for s in organic for c in s["assessment"]["claims"]) for v in VERDICTS},
        "llm_stance_from_analyze_v2_prompt": summarize(labels, llm_stance),
        "score_bias_audit_by_declared_stance": analytics.score_bias_audit(
            [s["assessment"]["score"]["final"] for s in organic], labels),
        "pipeline_latency_ms_per_opinion_median": float(np.median([s["assessment"]["latency_ms"] for s in scored])),
    })


def suite_stance(batch_size: int = 20) -> None:
    """Zero-shot LLM stance with the protesters' demand stated explicitly, against the audited labels."""
    records = [r for r in load_jsonl(CORPUS_DIR / "opinions.jsonl") if not r.get("campaign_id")]
    llm = LLMClient(api_key=secret("GROQ_API_KEY"), rate_limiter=TokenRateLimiter(7000))
    preds: dict[str, str] = {}
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        user = "\n\n".join(f'id={r["spec_id"]} | protesters\' demand: {prompts.protest_demand(r["protest_id"], r["protest_title"])}'
                           f'\ncomment: {r["text"]}' for r in batch)
        try:
            out = llm.json(system=prompts.STANCE_SYSTEM, user=user, schema=BatchStance, model=MODELS.analyze,
                           purpose="stance", max_tokens=1500)
        except LLMError as exc:
            print(f"stopping early at {len(preds)} opinions: {str(exc)[:200]}")
            break
        preds.update({p.id: p.stance for p in out.predictions})
    scored = [r for r in records if r["spec_id"] in preds]
    calls = list(llm.records)
    write("stance_llm", {
        "model": MODELS.analyze, "prompt": prompts.STANCE_VERSION, "n": len(scored),
        "vs_audited_labels": summarize([r["stance"] for r in scored], [preds[r["spec_id"]] for r in scored]),
        "vs_original_generator_labels": summarize([r.get("stance_generator", r["stance"]) for r in scored],
                                                  [preds[r["spec_id"]] for r in scored]),
        "tokens_per_opinion": round(sum(c.total_tokens for c in calls) / max(len(scored), 1), 1),
    })


def suite_summary() -> None:
    merged = {p.stem: json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json")) if p.stem != "summary"}
    for suite in merged.values():
        suite.pop("predictions", None)
        for cfg in suite.get("configs", {}).values():
            cfg.pop("per_claim", None)
        suite.pop("scores", None)
    (RESULTS / "summary.json").write_text(json.dumps(merged, indent=2, ensure_ascii=False, default=str))
    print(f"summary.json written with suites: {list(merged)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", choices=["retrieval", "paraphrase", "verify", "stance", "consistency", "endtoend", "summary"])
    parser.add_argument("--configs", nargs="*", default=["hybrid_k3__gpt-oss-120b", "hybrid_k3__gpt-oss-20b",
                                                         "bm25_k3__gpt-oss-120b", "oracle__gpt-oss-120b"])
    args = parser.parse_args()
    if args.suite == "retrieval":
        suite_retrieval()
    elif args.suite == "stance":
        suite_stance()
    elif args.suite == "paraphrase":
        suite_retrieval(paraphrase=True)
    elif args.suite == "verify":
        suite_verify(args.configs)
    elif args.suite == "consistency":
        suite_consistency()
    elif args.suite == "endtoend":
        suite_endtoend()
    else:
        suite_summary()


if __name__ == "__main__":
    main()
