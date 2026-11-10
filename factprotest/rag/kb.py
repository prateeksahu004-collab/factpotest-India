"""Knowledge-base chunks: Wikipedia sections split into overlapping passages."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
SKIP_SECTIONS = {
    "references", "external links", "see also", "further reading", "notes", "bibliography",
    "sources", "citations", "gallery", "footnotes", "explanatory notes",
}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    protest_id: int
    title: str
    section: str
    url: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.title} § {self.section}" if self.section else self.title


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split MediaWiki plain-text extracts on == headings ==, dropping reference-style sections."""
    sections: list[tuple[str, str]] = []
    path: list[str] = []
    last_end, current = 0, "Introduction"
    for match in HEADING_RE.finditer(text):
        sections.append((current, text[last_end:match.start()]))
        level = len(match.group(1)) - 2
        path = path[:level] + [match.group(2).strip()]
        current = " › ".join(path)
        last_end = match.end()
    sections.append((current, text[last_end:]))
    return [
        (name, body.strip()) for name, body in sections
        if body.strip() and name.split(" › ")[0].lower() not in SKIP_SECTIONS
    ]


def chunk_text(text: str, target_words: int = 160, overlap_words: int = 35) -> list[str]:
    sentences = [s.strip() for para in text.split("\n") for s in SENTENCE_RE.split(para) if s.strip()]
    chunks: list[str] = []
    window: list[str] = []
    count = 0
    for sentence in sentences:
        words = len(sentence.split())
        if window and count + words > target_words:
            chunks.append(" ".join(window))
            carry: list[str] = []
            carried = 0
            for prev in reversed(window):
                if carried >= overlap_words:
                    break
                carry.insert(0, prev)
                carried += len(prev.split())
            window, count = carry, carried
        window.append(sentence)
        count += words
    if window and (not chunks or count > overlap_words):
        chunks.append(" ".join(window))
    return [c for c in chunks if len(c.split()) >= 25]


def save_kb(chunks: list[Chunk], embeddings: np.ndarray, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "chunks.jsonl").open("w") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    np.save(directory / "embeddings.npy", embeddings.astype(np.float32))


def load_kb(directory: Path) -> tuple[list[Chunk], np.ndarray]:
    with (directory / "chunks.jsonl").open() as f:
        chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
    embeddings = np.load(directory / "embeddings.npy")
    if len(chunks) != len(embeddings):
        raise ValueError(f"KB mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")
    return chunks, embeddings
