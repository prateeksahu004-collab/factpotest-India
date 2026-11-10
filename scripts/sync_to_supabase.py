"""Replace the v1 template seed data in Supabase with the v2 corpus, and keep scores in sync.

Dry run by default. Modes:
  --apply          insert all corpus opinions (scored ones with their assessment, the rest marked synthetic and
                   unscored), verify the inserts, and only then delete the 1,000 template-generated v1 rows.
  --update-scores  copy assessments from opinions_scored.jsonl onto corpus rows already in Supabase that have none
                   (run after score_corpus.py has scored more of the corpus).
  --rescore-real   re-score real user submissions that still carry a v1 score (needs LLM quota).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factprotest.config import CORPUS_DIR, KB_DIR, secret  # noqa: E402
from factprotest.data import SupabaseRepository, parse_analysis, to_db_row  # noqa: E402

SEED_BATCH_PREFIX = "2026-10-31T18:16"


def load_corpus() -> list[dict[str, Any]]:
    scored_path = CORPUS_DIR / "opinions_scored.jsonl"
    scored = {r["id"]: r["assessment"] for r in map(json.loads, scored_path.open())} if scored_path.exists() else {}
    return [{**r, "assessment": scored.get(r["id"])} for r in map(json.loads, (CORPUS_DIR / "opinions.jsonl").open())]


def corpus_key(text: str, author: str) -> tuple[str, str]:
    return text.strip(), author.strip()


def classify(rows: list[dict[str, Any]]) -> tuple[list, list, list]:
    legacy = [r for r in rows if r["created_at"].startswith(SEED_BATCH_PREFIX) and parse_analysis(r).get("schema_version") != 2]
    synthetic = [r for r in rows if parse_analysis(r).get("source") == "synthetic"]
    real = [r for r in rows if r not in legacy and r not in synthetic]
    return legacy, synthetic, real


def apply_migration(repo: SupabaseRepository, corpus: list[dict[str, Any]], legacy: list, synthetic: list) -> None:
    if synthetic:
        raise SystemExit(f"{len(synthetic)} v2 corpus rows already exist; use --update-scores instead of --apply.")
    payload = [{k: v for k, v in to_db_row(r).items() if k != "id"} for r in corpus]
    for start in range(0, len(payload), 100):
        repo.sb.table("opinions").insert(payload[start:start + 100]).execute()
    _, inserted, _ = classify(repo.opinions())
    if len(inserted) != len(payload):
        raise SystemExit(f"Inserted {len(inserted)} of {len(payload)} rows; legacy rows were NOT deleted. Check and retry.")
    print(f"Inserted {len(inserted)} corpus rows.")
    for start in range(0, len(legacy), 200):
        repo.sb.table("opinions").delete().in_("id", [r["id"] for r in legacy[start:start + 200]]).execute()
    remaining, _, _ = classify(repo.opinions())
    if remaining:
        print(f"WARNING: {len(remaining)} legacy rows could not be deleted (row-level security?). Delete them in the "
              "Supabase table editor: opinions created at 2026-10-31 18:16.")
    else:
        print(f"Deleted {len(legacy)} legacy template rows.")


def update_scores(repo: SupabaseRepository, corpus: list[dict[str, Any]], synthetic: list) -> int:
    by_key = {corpus_key(r["text"], r["author"]): r for r in corpus if r["assessment"]}
    updated = 0
    for row in synthetic:
        record = by_key.get(corpus_key(row["opinion_text"], row["user_name"]))
        if record and parse_analysis(row).get("schema_version") != 2:
            fresh = to_db_row(record)
            repo.sb.table("opinions").update({"credibility_score": fresh["credibility_score"],
                                              "ai_analysis": fresh["ai_analysis"]}).eq("id", row["id"]).execute()
            updated += 1
    return updated


def rescore_real(repo: SupabaseRepository, real: list[dict[str, Any]]) -> None:
    from factprotest.factcheck import OpinionInput, assess
    from factprotest.llm import LLMClient, TokenRateLimiter
    from factprotest.rag.embed import get_embedder
    from factprotest.rag.kb import load_kb
    from factprotest.rag.retriever import HybridRetriever

    chunks, embeddings = load_kb(KB_DIR)
    llm = LLMClient(api_key=secret("GROQ_API_KEY"), rate_limiter=TokenRateLimiter(7000))
    todo = [r for r in real if parse_analysis(r).get("schema_version") != 2]
    items = [OpinionInput(str(r["id"]), int(r["issue_id"]), r["issue_title"], r["stance"], r["opinion_text"]) for r in todo]
    results = assess(llm, HybridRetriever(chunks, embeddings, get_embedder()), items)
    for r in todo:
        if (a := results.get(str(r["id"]))):
            repo.sb.table("opinions").update({"credibility_score": a.score.final,
                                              "ai_analysis": a.model_dump_json()}).eq("id", r["id"]).execute()
    print(f"Re-scored {len(results)} of {len(todo)} real submissions.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--update-scores", action="store_true")
    parser.add_argument("--rescore-real", action="store_true")
    args = parser.parse_args()

    repo = SupabaseRepository()
    corpus = load_corpus()
    legacy, synthetic, real = classify(repo.opinions())
    scored = sum(r["assessment"] is not None for r in corpus)
    print(f"Supabase now: {len(legacy)} legacy template rows, {len(synthetic)} v2 corpus rows, {len(real)} real submissions")
    print(f"Local corpus: {len(corpus)} opinions ({scored} fact-checked, {len(corpus) - scored} queued)")
    if args.apply:
        apply_migration(repo, corpus, legacy, synthetic)
    if args.update_scores:
        print(f"Updated scores on {update_scores(repo, corpus, classify(repo.opinions())[1])} rows.")
    if args.rescore_real:
        rescore_real(repo, real)
    if not (args.apply or args.update_scores or args.rescore_real):
        print(f"Dry run. --apply would insert {len(corpus)} rows, then delete {len(legacy)} legacy rows; "
              f"{len(real)} real submissions are kept.")


if __name__ == "__main__":
    main()
