"""Generate a persona-conditioned synthetic opinion corpus with controlled ground truth.

Every opinion is generated from an explicit spec (stance, writing style, state, factual mode). Ground truth
that real data would not give us is recorded alongside the text:
  * stance labels         -> train/evaluate the stance classifier
  * injected false claims -> measure end-to-end misinformation recall of the fact-check pipeline
  * coordinated campaigns -> measure precision/recall of the inauthentic-behaviour detector
The corpus is labelled `source: synthetic` everywhere it is shown.
"""
from __future__ import annotations

import json
import random
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factprotest import prompts  # noqa: E402
from factprotest.config import CORPUS_DIR, DATA_DIR, KB_DIR, MODELS, secret  # noqa: E402
from factprotest.llm import LLMClient, TokenRateLimiter  # noqa: E402
from factprotest.rag.kb import load_kb  # noqa: E402

PER_PROTEST = 40
BATCH = 10
SEED = 42
OUT = CORPUS_DIR / "opinions.jsonl"
NAMESPACE = uuid.UUID("6f1c2b1e-9a57-4f0e-8b3c-2d9a1e4f7c10")
PLATFORM_START = datetime(2026, 10, 31, 9, 0, tzinfo=UTC)

STANCE_MIX = {  # (support, oppose, neutral): deliberately varied so polarization differs across protests
    1: (0.55, 0.30, 0.15), 2: (0.75, 0.08, 0.17), 3: (0.45, 0.40, 0.15), 4: (0.65, 0.20, 0.15),
    5: (0.60, 0.15, 0.25), 6: (0.45, 0.45, 0.10), 7: (0.80, 0.05, 0.15), 8: (0.60, 0.15, 0.25),
    9: (0.40, 0.40, 0.20), 10: (0.50, 0.30, 0.20),
}
CORE_STATES = {
    1: ["Punjab", "Haryana", "Uttar Pradesh", "Delhi"], 2: ["Bihar", "Rajasthan", "Uttar Pradesh", "Maharashtra", "Delhi"],
    3: ["Bihar", "Uttar Pradesh", "Haryana", "Madhya Pradesh"], 4: ["Haryana", "Delhi", "Uttar Pradesh"],
    5: ["Ladakh", "Jammu and Kashmir", "Delhi", "Karnataka"], 6: ["Assam", "West Bengal", "Delhi", "Uttar Pradesh", "Kerala"],
    7: ["Delhi", "Haryana", "Punjab", "Uttar Pradesh"], 8: ["Manipur", "Assam", "Delhi", "Mizoram"],
    9: ["Gujarat", "Maharashtra", "Delhi", "Karnataka"], 10: ["Maharashtra"],
}
ALL_STATES = ["Andhra Pradesh", "Assam", "Bihar", "Delhi", "Goa", "Gujarat", "Haryana", "Jharkhand", "Karnataka",
              "Kerala", "Madhya Pradesh", "Maharashtra", "Odisha", "Punjab", "Rajasthan", "Tamil Nadu", "Telangana",
              "Uttar Pradesh", "Uttarakhand", "West Bengal", "Not Specified"]
STYLES = [
    ("formal and analytical", "60-110 words"), ("casual social-media post", "20-50 words"),
    ("Hinglish: Hindi words in Latin script mixed with English", "25-60 words"),
    ("short, angry one-liner", "8-20 words"), ("personal anecdote from lived experience", "50-90 words"),
    ("sarcastic", "20-45 words"), ("measured and balanced", "40-80 words"),
]
PERSONAS = ["college student", "retired government employee", "small farmer", "software engineer", "homemaker",
            "school teacher", "auto-rickshaw driver", "small business owner", "journalism student", "doctor",
            "ex-serviceman", "lawyer", "daily-wage worker", "civil-services aspirant", "nurse", "shopkeeper"]
FIRST = ["Aarav", "Ananya", "Rohit", "Priya", "Imran", "Fatima", "Gurpreet", "Harleen", "Joseph", "Mary", "Lalthanzuala",
         "Thoibi", "Venkatesh", "Lakshmi", "Arjun", "Sneha", "Tenzin", "Dolkar", "Suresh", "Kavya", "Abdul", "Zoya",
         "Rakesh", "Pooja", "Manoj", "Divya", "Sourav", "Moumita", "Karthik", "Meenakshi", "Vikram", "Nikita"]
LAST = ["Sharma", "Singh", "Khan", "Gill", "Thomas", "Reddy", "Iyer", "Das", "Yadav", "Patil", "Meitei", "Haokip",
        "Namgyal", "Choudhary", "Mishra", "Nair", "Banerjee", "Qureshi", "Joshi", "Rao", "Kumar", "Verma", "Ansari"]
CAMPAIGNS = [
    {"campaign_id": "C1", "protest_id": 6, "stance": "Oppose", "tag": "#SupportCAA",
     "seed": "The CAA does not take away anyone's citizenship. These protests were funded by outside forces and ordinary "
             "people were misled by propaganda. Read the Act before you protest."},
    {"campaign_id": "C2", "protest_id": 9, "stance": "Oppose", "tag": "#IndiaFirst",
     "seed": "Hindenburg is a foreign short-seller that profits when Indian companies fall. The demand for a JPC probe is "
             "politically motivated and hurts India's growth story."},
    {"campaign_id": "C3", "protest_id": 3, "stance": "Support", "tag": "#ScrapAgnipath",
     "seed": "Agnipath betrays the youth of this country. Four years of service with no pension is unacceptable for those "
             "who want to serve. Roll it back now."},
]
CAMPAIGN_VARIANTS = 7
SYNONYMS = {"protests": "agitations", "funded": "sponsored", "ordinary people": "common citizens", "misled": "fooled",
            "propaganda": "fake narratives", "profits": "makes money", "fall": "crash", "politically motivated": "purely political",
            "hurts": "damages", "betrays": "cheats", "unacceptable": "not acceptable", "Roll it back now": "Withdraw it immediately",
            "anyone's": "any Indian's", "growth story": "economic rise"}
OPENERS = ["", "", "Truth: ", "Wake up! ", "Exactly this 👇 ", "Simple facts. "]
CLOSERS = ["", " Share this.", " 🇮🇳", " Jai Hind!", " Spread the word."]


def campaign_variant(seed: str, tag: str, rng: random.Random) -> str:
    """Copy-paste campaign edits: a few synonym swaps, optional sentence rotation, opener/closer and hashtags."""
    text = seed
    for old, new in rng.sample(sorted(SYNONYMS.items()), k=len(SYNONYMS)):
        if old in text and rng.random() < 0.35:
            text = text.replace(old, new)
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if len(sentences) > 2 and rng.random() < 0.3:
        sentences = sentences[1:] + sentences[:1]
    text = ". ".join(s.rstrip(".") for s in sentences) + "."
    hashtags = " ".join(rng.sample([tag, "#Truth", "#India", "#WakeUpIndia"], k=rng.randint(0, 2)))
    return f"{rng.choice(OPENERS)}{text}{rng.choice(CLOSERS)} {hashtags}".strip()


class GeneratedOpinion(BaseModel):
    id: str
    text: str
    false_claim: str | None = None
    true_version: str | None = None


class GeneratedBatch(BaseModel):
    opinions: list[GeneratedOpinion]



def make_specs(rng: random.Random, protest_id: int) -> list[dict]:
    support, oppose, neutral = STANCE_MIX[protest_id]
    n_support, n_oppose = round(PER_PROTEST * support), round(PER_PROTEST * oppose)
    stances = ["Support"] * n_support + ["Oppose"] * n_oppose + ["Neutral"] * (PER_PROTEST - n_support - n_oppose)
    rng.shuffle(stances)
    specs = []
    for i, stance in enumerate(stances):
        style, length = rng.choice(STYLES)
        mode = rng.choices(["accurate_fact", "false_fact", "no_fact"], weights=[40, 25, 35])[0]
        state = rng.choice(CORE_STATES[protest_id]) if rng.random() < 0.6 else rng.choice(ALL_STATES)
        specs.append({"id": f"p{protest_id}-{i:02d}", "stance": stance, "style": style, "length": length,
                      "factual_mode": mode, "state": state, "persona": f"{rng.randint(19, 68)}-year-old {rng.choice(PERSONAS)}",
                      "author": f"{rng.choice(FIRST)} {rng.choice(LAST)}"})
    return specs


def passages_block(chunks: list, rng: random.Random, k: int = 4) -> str:
    picked = rng.sample(chunks, min(k, len(chunks)))
    return "\n\n".join(f"[{c.chunk_id}] {' '.join(c.text.split()[:130])}" for c in picked)


def timestamp(rng: random.Random) -> str:
    return (PLATFORM_START + timedelta(minutes=rng.randint(0, 9 * 24 * 60))).isoformat()


def main() -> None:
    rng = random.Random(SEED)
    protests = {p["id"]: p for p in json.loads((DATA_DIR / "protests.json").read_text())}
    chunks, _ = load_kb(KB_DIR)
    llm = LLMClient(api_key=secret("GROQ_API_KEY"), rate_limiter=TokenRateLimiter(7000))
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["spec_id"] for line in OUT.open()} if OUT.exists() else set()

    with OUT.open("a") as out:
        for protest_id, protest in protests.items():
            specs = make_specs(rng, protest_id)
            pchunks = [c for c in chunks if c.protest_id == protest_id]
            for start in range(0, len(specs), BATCH):
                batch = specs[start:start + BATCH]
                passages = passages_block(pchunks, rng)
                created = [timestamp(rng) for _ in batch]
                if all(s["id"] in done for s in batch):
                    continue
                spec_lines = "\n".join(
                    f'- id={s["id"]} | stance={s["stance"]} | persona={s["persona"]} from {s["state"]} | '
                    f'style={s["style"]} | length={s["length"]} | factual_mode={s["factual_mode"]}' for s in batch)
                user = (f"Protest: {protest['title']}\nBackground: {protest['description']}\n"
                        f"Outcome so far: {protest.get('outcome') or 'ongoing'}\n\nReference passages:\n{passages}\n\n"
                        f"Write one comment per spec:\n{spec_lines}")
                result = llm.json(system=prompts.CORPUS_SYSTEM, user=user, schema=GeneratedBatch,
                                  model=MODELS.generator, purpose="corpus", temperature=0.9, max_tokens=4000)
                by_id = {o.id: o for o in result.opinions}
                for spec, ts in zip(batch, created):
                    gen = by_id.get(spec["id"])
                    if gen is None or spec["id"] in done:
                        continue
                    false_claim = gen.false_claim if spec["factual_mode"] == "false_fact" else None
                    record = {
                        "id": str(uuid.uuid5(NAMESPACE, spec["id"])), "spec_id": spec["id"], "protest_id": protest_id,
                        "protest_title": protest["title"], "author": spec["author"], "state": spec["state"],
                        "stance": spec["stance"], "style": spec["style"], "persona": spec["persona"],
                        "factual_mode": spec["factual_mode"], "false_claim": false_claim,
                        "true_version": gen.true_version if false_claim else None, "campaign_id": None,
                        "text": gen.text.strip(), "created_at": ts, "source": "synthetic",
                        "generator": {"model": MODELS.generator, "prompt": prompts.CORPUS_VERSION},
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                print(f"P{protest_id} batch {start // BATCH + 1}: {len(result.opinions)} opinions")

        for campaign in CAMPAIGNS:
            seed_id = f"{campaign['campaign_id']}-00"
            if seed_id in done:
                continue
            protest = protests[campaign["protest_id"]]
            posts = [campaign["seed"]] + [campaign_variant(campaign["seed"], campaign["tag"], rng)
                                          for _ in range(CAMPAIGN_VARIANTS)]
            burst_start = PLATFORM_START + timedelta(days=rng.randint(2, 7), hours=rng.randint(0, 12))
            for i, text in enumerate(posts):
                spec_id = f"{campaign['campaign_id']}-{i:02d}"
                record = {
                    "id": str(uuid.uuid5(NAMESPACE, spec_id)), "spec_id": spec_id, "protest_id": campaign["protest_id"],
                    "protest_title": protest["title"], "author": f"{rng.choice(FIRST)}_{rng.randint(100, 9999)}",
                    "state": rng.choice(ALL_STATES), "stance": campaign["stance"], "style": "coordinated campaign",
                    "persona": "campaign account", "factual_mode": "no_fact", "false_claim": None, "true_version": None,
                    "campaign_id": campaign["campaign_id"], "text": text.strip(),
                    "created_at": (burst_start + timedelta(minutes=rng.randint(0, 90))).isoformat(),
                    "source": "synthetic", "generator": {"model": "rule-based copy-paste variants", "prompt": None},
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"Campaign {campaign['campaign_id']}: {len(posts)} posts")

    print(f"Corpus: {sum(1 for _ in OUT.open())} opinions; LLM usage: {llm.usage_summary()}")


if __name__ == "__main__":
    main()
