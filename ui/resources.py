"""Cached, process-wide resources for the Streamlit app."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import numpy as np
import streamlit as st

from factprotest.agent import AnalystContext, OpinionIndex
from factprotest.config import ARTIFACTS_DIR, EVAL_DIR, KB_DIR, secret
from factprotest.data import LocalRepository, Repository, SupabaseRepository
from factprotest.llm import LLMClient
from factprotest.ml.integrity import detect_coordination
from factprotest.ml.stance import StanceModel
from factprotest.rag.embed import get_embedder
from factprotest.rag.kb import load_kb
from factprotest.rag.retriever import HybridRetriever

COORDINATION_THRESHOLD = 0.88


def data_mode() -> str:
    return os.environ.get("FACTPROTEST_DATA", "supabase")


@st.cache_resource(show_spinner=False)
def repo() -> Repository:
    return LocalRepository() if data_mode() == "local" else SupabaseRepository()


@st.cache_resource(show_spinner=False)
def llm() -> LLMClient:
    return LLMClient(api_key=secret("GROQ_API_KEY"), max_retries=3)


@st.cache_resource(show_spinner="Loading knowledge base and embedding model…")
def retriever() -> HybridRetriever:
    chunks, embeddings = load_kb(KB_DIR)
    return HybridRetriever(chunks, embeddings, get_embedder())


@st.cache_resource(show_spinner=False)
def stance_model() -> StanceModel | None:
    path = ARTIFACTS_DIR / "stance_model.joblib"
    return StanceModel.load(path) if path.exists() else None


def _read_json(path: Any, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


@st.cache_data(show_spinner=False)
def topics() -> dict[str, Any]:
    return _read_json(ARTIFACTS_DIR / "topics.json", {})


@st.cache_data(show_spinner=False)
def eval_summary() -> dict[str, Any]:
    return _read_json(EVAL_DIR / "results" / "summary.json", {})


@st.cache_data(show_spinner=False)
def ml_metrics() -> dict[str, Any]:
    return {name: _read_json(ARTIFACTS_DIR / "metrics" / f"{name}.json", {}) for name in ("stance", "integrity")}


@st.cache_data(ttl=60, show_spinner=False)
def protests() -> list[dict[str, Any]]:
    return repo().protests()


@st.cache_data(ttl=30, show_spinner=False)
def opinions() -> list[dict[str, Any]]:
    return repo().opinions()


@st.cache_data(ttl=60, show_spinner=False)
def replies() -> list[dict[str, Any]]:
    return repo().replies()


def refresh() -> None:
    opinions.clear()
    replies.clear()
    protests.clear()


@st.cache_resource(show_spinner="Indexing opinions…", max_entries=3)
def _opinion_vectors(key: str, texts: tuple[str, ...]) -> np.ndarray:
    return get_embedder().documents(list(texts))


def opinion_index(rows: list[dict[str, Any]]) -> OpinionIndex:
    texts = tuple(o["opinion_text"] for o in rows)
    key = hashlib.sha1("|".join(str(o["id"]) for o in rows).encode()).hexdigest()
    embedder = get_embedder()
    return OpinionIndex([str(o["id"]) for o in rows], _opinion_vectors(key, texts),
                        lambda q: embedder.queries([q])[0])


def coordination(rows: list[dict[str, Any]], index: OpinionIndex) -> list[dict[str, Any]]:
    """Run the coordinated-campaign detector live over whatever opinions are currently stored."""
    records = [{"id": o["id"], "protest_id": int(o["issue_id"]), "created_at": o["created_at"],
                "author": o.get("user_name", ""), "stance": o["stance"]} for o in rows]
    return [c.__dict__ for c in detect_coordination(records, index.embeddings, threshold=COORDINATION_THRESHOLD)]


def analyst_context() -> AnalystContext:
    rows = opinions()
    index = opinion_index(rows)
    return AnalystContext(protests=protests(), opinions=rows, replies=replies(), retriever=retriever(),
                          opinion_index=index, topics=topics(), coordinated=coordination(rows, index))
