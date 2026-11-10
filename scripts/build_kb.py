"""Build the fact-checking knowledge base from Wikipedia (CC BY-SA 4.0).

Fetches plain-text extracts for each protest's source articles, splits them by section into
~160-word overlapping passages, embeds them with bge-small-en-v1.5 and writes data/kb/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factprotest.config import DATA_DIR, KB_DIR  # noqa: E402
from factprotest.data import fetch_protests_from_supabase  # noqa: E402
from factprotest.rag.embed import Embedder  # noqa: E402
from factprotest.rag.kb import Chunk, chunk_text, save_kb, split_sections  # noqa: E402

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "FactProtestIndia/2.0 (civic-tech research; https://github.com/prateeksahu004-collab/factpotest-India)"}
MAX_CHUNKS_PER_ARTICLE = 150

SOURCES: dict[int, list[str]] = {
    1: ["2020–2021 Indian farmers' protest", "2020 Indian agriculture acts"],
    2: ["2024 NEET controversy", "Indian examination paper leaks"],
    3: ["Agnipath Scheme"],
    4: ["2023 Indian wrestlers' protest", "Brij Bhushan Sharan Singh"],
    5: ["Ladakh protests", "Sonam Wangchuk"],
    6: ["Citizenship Amendment Act protests", "Citizenship (Amendment) Act, 2019"],
    7: ["Air pollution in Delhi", "Environmental issues in Delhi"],
    8: ["Manipur conflict (2023–present)"],
    9: ["Hindenburg Research", "Adani Group"],
    10: ["Brihanmumbai Electric Supply and Transport"],
}


def fetch_article(title: str) -> dict:
    params = {
        "action": "query", "format": "json", "formatversion": 2, "redirects": 1, "titles": title,
        "prop": "extracts|info|revisions", "explaintext": 1, "exsectionformat": "wiki",
        "inprop": "url", "rvprop": "ids|timestamp",
    }
    response = requests.get(API, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    page = response.json()["query"]["pages"][0]
    if page.get("missing"):
        raise ValueError(f"Wikipedia article not found: {title}")
    revision = page["revisions"][0]
    return {"title": page["title"], "url": page["fullurl"], "revid": revision["revid"],
            "timestamp": revision["timestamp"], "text": page["extract"]}


def main() -> None:
    protests = fetch_protests_from_supabase()
    (DATA_DIR / "protests.json").write_text(json.dumps(protests, indent=2, ensure_ascii=False))
    print(f"Saved {len(protests)} protests to data/protests.json")

    chunks: list[Chunk] = []
    manifest = []
    for protest_id, titles in SOURCES.items():
        for title in titles:
            article = fetch_article(title)
            slug = "".join(ch for ch in article["title"].lower() if ch.isalnum())[:24]
            article_chunks: list[Chunk] = []
            for section, body in split_sections(article["text"]):
                for passage in chunk_text(body):
                    article_chunks.append(Chunk(
                        chunk_id=f"P{protest_id}-{slug}-{len(article_chunks):03d}", protest_id=protest_id,
                        title=article["title"], section=section, url=article["url"], text=passage,
                    ))
            article_chunks = article_chunks[:MAX_CHUNKS_PER_ARTICLE]
            chunks.extend(article_chunks)
            manifest.append({"protest_id": protest_id, "title": article["title"], "url": article["url"],
                             "revid": article["revid"], "revision_timestamp": article["timestamp"],
                             "words": len(article["text"].split()), "chunks": len(article_chunks)})
            print(f"  P{protest_id} {article['title']}: {len(article_chunks)} chunks")
            time.sleep(0.5)

    print(f"Embedding {len(chunks)} chunks ...")
    start = time.perf_counter()
    embeddings = Embedder().documents([f"{c.title}. {c.section}. {c.text}" for c in chunks])
    print(f"  done in {time.perf_counter() - start:.1f}s, shape={embeddings.shape}")
    save_kb(chunks, embeddings, KB_DIR)
    (KB_DIR / "sources.json").write_text(json.dumps({
        "license": "Text from Wikipedia, CC BY-SA 4.0. Revision ids pinned for reproducibility.",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "articles": manifest,
    }, indent=2, ensure_ascii=False))
    print(f"Wrote {KB_DIR}")


if __name__ == "__main__":
    main()
