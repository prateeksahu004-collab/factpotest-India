"""Run the full production pipeline (safety -> claim extraction -> retrieval -> verification -> scoring) over
the opinion corpus. Resumable: already-scored ids are skipped, so it can be re-run after interruptions."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factprotest import safety  # noqa: E402
from factprotest.config import CORPUS_DIR, KB_DIR, secret  # noqa: E402
from factprotest.factcheck import OpinionInput, assess  # noqa: E402
from factprotest.llm import LLMClient, LLMError, TokenRateLimiter  # noqa: E402
from factprotest.rag.embed import get_embedder  # noqa: E402
from factprotest.rag.kb import load_kb  # noqa: E402
from factprotest.rag.retriever import HybridRetriever  # noqa: E402

SRC = CORPUS_DIR / "opinions.jsonl"
OUT = CORPUS_DIR / "opinions_scored.jsonl"
BATCH = 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-moderation", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")

    records = [json.loads(line) for line in SRC.open()]
    done = {json.loads(line)["id"] for line in OUT.open()} if OUT.exists() else set()
    todo = [r for r in records if r["id"] not in done][: args.limit]
    print(f"{len(records)} in corpus, {len(done)} already scored, {len(todo)} to score")

    chunks, embeddings = load_kb(KB_DIR)
    retriever = HybridRetriever(chunks, embeddings, get_embedder())
    llm = LLMClient(api_key=secret("GROQ_API_KEY"), rate_limiter=TokenRateLimiter(7000))
    start = time.perf_counter()
    with OUT.open("a") as out:
        for i in range(0, len(todo), BATCH):
            batch = todo[i:i + BATCH]
            items = []
            for r in batch:
                item = OpinionInput(r["id"], r["protest_id"], r["protest_title"], r["stance"], r["text"])
                item.safety = safety.check(llm, r["text"], moderation=not args.no_moderation)
                items.append(item)
            try:
                results = assess(llm, retriever, items)
            except LLMError as exc:
                print(f"  batch {i // BATCH + 1} failed: {exc}; will be retried on next run")
                continue
            for r in batch:
                if r["id"] in results:
                    out.write(json.dumps({**r, "assessment": results[r["id"]].model_dump(),
                                          "scored_at": datetime.now(UTC).isoformat()}, ensure_ascii=False) + "\n")
            out.flush()
            elapsed = time.perf_counter() - start
            print(f"  {min(i + BATCH, len(todo))}/{len(todo)} scored  ({elapsed / 60:.1f} min)  {llm.usage_summary()}")
    print("done", llm.usage_summary())


if __name__ == "__main__":
    main()
