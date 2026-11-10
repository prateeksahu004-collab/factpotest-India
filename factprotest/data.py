"""Repository layer. Supabase in production; a read-mostly local JSONL backend for offline dev, tests and CI."""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from factprotest.config import CORPUS_DIR, DATA_DIR, secret


class Repository(Protocol):
    def protests(self) -> list[dict[str, Any]]: ...
    def opinions(self, protest_id: int | None = None) -> list[dict[str, Any]]: ...
    def add_opinion(self, row: dict[str, Any]) -> dict[str, Any]: ...
    def replies(self, protest_title: str | None = None) -> list[dict[str, Any]]: ...
    def add_reply(self, row: dict[str, Any]) -> None: ...
    def reactions(self, protest_id: int) -> tuple[int, int]: ...
    def add_reaction(self, protest_id: int, reaction: str) -> None: ...


def _supabase_client() -> Any:
    from supabase import create_client

    return create_client(secret("SUPABASE_URL"), secret("SUPABASE_KEY"))


def fetch_protests_from_supabase() -> list[dict[str, Any]]:
    return _supabase_client().table("protests").select("*").order("id").execute().data or []


class SupabaseRepository:
    def __init__(self, client: Any | None = None):
        self.sb = client or _supabase_client()

    def protests(self) -> list[dict[str, Any]]:
        return self.sb.table("protests").select("*").order("id").execute().data or []

    def opinions(self, protest_id: int | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        page = 1000
        while True:
            query = self.sb.table("opinions").select("*").order("created_at", desc=True)
            if protest_id is not None:
                query = query.eq("issue_id", str(protest_id))
            batch = query.range(len(rows), len(rows) + page - 1).execute().data or []
            rows.extend(batch)
            if len(batch) < page:
                return rows

    def add_opinion(self, row: dict[str, Any]) -> dict[str, Any]:
        return (self.sb.table("opinions").insert(row).execute().data or [row])[0]

    def replies(self, protest_title: str | None = None) -> list[dict[str, Any]]:
        query = self.sb.table("authority_replies").select("*").order("created_at", desc=True)
        if protest_title:
            query = query.eq("issue_title", protest_title)
        return query.execute().data or []

    def add_reply(self, row: dict[str, Any]) -> None:
        self.sb.table("authority_replies").insert(row).execute()

    def reactions(self, protest_id: int) -> tuple[int, int]:
        data = self.sb.table("citizen_reactions").select("reaction").eq("issue_id", str(protest_id)).execute().data or []
        return sum(r["reaction"] == "up" for r in data), sum(r["reaction"] == "down" for r in data)

    def add_reaction(self, protest_id: int, reaction: str) -> None:
        self.sb.table("citizen_reactions").insert({"issue_id": str(protest_id), "reaction": reaction}).execute()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


class LocalRepository:
    """Serves data/protests.json and the research corpus (with pipeline assessments where scored); writes stay in memory."""

    def __init__(self, protests_path: Path = DATA_DIR / "protests.json",
                 corpus_path: Path = CORPUS_DIR / "opinions.jsonl",
                 scored_path: Path = CORPUS_DIR / "opinions_scored.jsonl"):
        self._protests = json.loads(protests_path.read_text()) if protests_path.exists() else []
        scored = {r["id"]: r["assessment"] for r in _read_jsonl(scored_path)}
        self._opinions = [to_db_row({**r, "assessment": scored.get(r["id"])}) for r in _read_jsonl(corpus_path)]
        self._replies: list[dict[str, Any]] = []
        self._reactions: list[tuple[int, str]] = []

    def protests(self) -> list[dict[str, Any]]:
        return self._protests

    def opinions(self, protest_id: int | None = None) -> list[dict[str, Any]]:
        rows = self._opinions if protest_id is None else [o for o in self._opinions if o["issue_id"] == str(protest_id)]
        return sorted(rows, key=lambda o: o["created_at"], reverse=True)

    def add_opinion(self, row: dict[str, Any]) -> dict[str, Any]:
        stored = {"id": str(uuid.uuid4()), "created_at": datetime.now(UTC).isoformat(), **row}
        self._opinions.append(stored)
        return stored

    def replies(self, protest_title: str | None = None) -> list[dict[str, Any]]:
        return [r for r in self._replies if protest_title in (None, r["issue_title"])]

    def add_reply(self, row: dict[str, Any]) -> None:
        self._replies.insert(0, {"id": str(uuid.uuid4()), "created_at": datetime.now(UTC).isoformat(), **row})

    def reactions(self, protest_id: int) -> tuple[int, int]:
        mine = [r for pid, r in self._reactions if pid == protest_id]
        return mine.count("up"), mine.count("down")

    def add_reaction(self, protest_id: int, reaction: str) -> None:
        self._reactions.append((protest_id, reaction))


def to_db_row(record: dict[str, Any]) -> dict[str, Any]:
    """Map a scored corpus record onto the Supabase `opinions` row shape."""
    assessment = record.get("assessment")
    return {
        "id": record["id"],
        "issue_id": str(record["protest_id"]),
        "issue_title": record["protest_title"],
        "user_name": record["author"],
        "state": record["state"],
        "stance": record["stance"],
        "opinion_text": record["text"],
        "credibility_score": assessment["score"]["final"] if assessment else None,
        "ai_analysis": json.dumps({**(assessment or {}), "source": "synthetic"}),
        "created_at": record["created_at"],
    }


def parse_analysis(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("ai_analysis")
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
