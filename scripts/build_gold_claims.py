"""Build the claim-verification gold set from knowledge-base passages.

For sampled passages a *different* model family (Qwen) than the verifier (gpt-oss) drafts one SUPPORTED,
one REFUTED (single-detail perturbation) and one NOT_ENOUGH_INFO claim, keeping the source passage id so
retrieval can be scored too. Drafts are then manually reviewed (see evals/README.md).
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factprotest import prompts  # noqa: E402
from factprotest.config import EVAL_DIR, KB_DIR, MODELS, secret  # noqa: E402
from factprotest.llm import LLMClient, TokenRateLimiter  # noqa: E402
from factprotest.rag.kb import load_kb  # noqa: E402

PASSAGES_PER_PROTEST = 4
PASSAGES_PER_CALL = 2
OUT = EVAL_DIR / "gold_claims_draft.jsonl"


class GoldItem(BaseModel):
    passage_id: str
    supported: str
    refuted: str
    not_enough_info: str


class GoldBatch(BaseModel):
    items: list[GoldItem]


def main() -> None:
    rng = random.Random(7)
    chunks, _ = load_kb(KB_DIR)
    llm = LLMClient(api_key=secret("GROQ_API_KEY"), rate_limiter=TokenRateLimiter(7000))
    rows = []
    for protest_id in sorted({c.protest_id for c in chunks}):
        candidates = [c for c in chunks if c.protest_id == protest_id and re.search(r"\d", c.text)
                      and len(c.text.split()) >= 80 and "Introduction" not in c.section] or \
                     [c for c in chunks if c.protest_id == protest_id]
        picked = rng.sample(candidates, min(PASSAGES_PER_PROTEST, len(candidates)))
        for start in range(0, len(picked), PASSAGES_PER_CALL):
            group = picked[start:start + PASSAGES_PER_CALL]
            user = "\n\n".join(f"[{c.chunk_id}] ({c.citation}) {c.text}" for c in group)
            batch = llm.json(system=prompts.GOLD_SYSTEM, user=user, schema=GoldBatch, model=MODELS.gold_generator,
                             purpose="gold", temperature=0.4, max_tokens=900)
            valid = {c.chunk_id for c in group}
            for item in batch.items:
                if item.passage_id not in valid:
                    continue
                for label, claim in (("SUPPORTED", item.supported), ("REFUTED", item.refuted),
                                     ("NOT_ENOUGH_INFO", item.not_enough_info)):
                    rows.append({"id": f"G{len(rows) + 1:03d}", "protest_id": protest_id, "claim": claim.strip(),
                                 "label": label, "source_chunk_id": item.passage_id,
                                 "generator": {"model": MODELS.gold_generator, "prompt": prompts.GOLD_VERSION}})
        print(f"P{protest_id}: {len(rows)} claims total")
    with OUT.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} draft claims to {OUT}; usage {llm.usage_summary()}")


if __name__ == "__main__":
    main()
