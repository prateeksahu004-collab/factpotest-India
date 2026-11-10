"""Replace the v1 template seed data in Supabase with the scored v2 corpus.

Dry run by default. With --apply it:
  1. deletes the 1,000 template-generated v1 rows (all inserted in one batch by the old bulk_insert.py),
  2. inserts the scored synthetic corpus (marked source=synthetic inside ai_analysis),
  3. re-scores any remaining real user submissions with the v2 pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factprotest.config import CORPUS_DIR, KB_DIR, secret  # noqa: E402
from factprotest.data import SupabaseRepository, parse_analysis, to_db_row  # noqa: E402
from factprotest.factcheck import OpinionInput, assess  # noqa: E402
from factprotest.llm import LLMClient, TokenRateLimiter  # noqa: E402
from factprotest.rag.embed import get_embedder  # noqa: E402
from factprotest.rag.kb import load_kb  # noqa: E402
from factprotest.rag.retriever import HybridRetriever  # noqa: E402

SEED_BATCH_PREFIX = "2026-10-31T18:16"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="actually write to Supabase")
    args = parser.parse_args()

    repo = SupabaseRepository()
    rows = repo.opinions()
    legacy = [r for r in rows if r["created_at"].startswith(SEED_BATCH_PREFIX) and parse_analysis(r).get("schema_version") != 2]
    real = [r for r in rows if r not in legacy and parse_analysis(r).get("schema_version") != 2
            and parse_analysis(r).get("source") != "synthetic"]
    corpus = [json.loads(line) for line in (CORPUS_DIR / "opinions_scored.jsonl").open()]
    print(f"Supabase: {len(rows)} opinions | legacy template rows to delete: {len(legacy)} | "
          f"real submissions to re-score: {len(real)} | scored corpus rows to insert: {len(corpus)}")
    if not args.apply:
        print("Dry run only. Re-run with --apply to write.")
        return

    for start in range(0, len(legacy), 200):
        ids = [r["id"] for r in legacy[start:start + 200]]
        repo.sb.table("opinions").delete().in_("id", ids).execute()
    remaining = len(repo.opinions())
    if remaining > len(rows) - len(legacy):
        raise SystemExit("Delete did not take effect (row-level security?). Nothing inserted.")

    payload = [{k: v for k, v in to_db_row(r).items() if k != "id"} for r in corpus]
    for start in range(0, len(payload), 100):
        repo.sb.table("opinions").insert(payload[start:start + 100]).execute()

    if real:
        chunks, embeddings = load_kb(KB_DIR)
        llm = LLMClient(api_key=secret("GROQ_API_KEY"), rate_limiter=TokenRateLimiter(7000))
        items = [OpinionInput(str(r["id"]), int(r["issue_id"]), r["issue_title"], r["stance"], r["opinion_text"]) for r in real]
        results = assess(llm, HybridRetriever(chunks, embeddings, get_embedder()), items)
        for r in real:
            a = results.get(str(r["id"]))
            if a:
                repo.sb.table("opinions").update({"credibility_score": a.score.final,
                                                  "ai_analysis": a.model_dump_json()}).eq("id", r["id"]).execute()
    print(f"Done: {len(repo.opinions())} opinions now in Supabase.")


if __name__ == "__main__":
    main()
