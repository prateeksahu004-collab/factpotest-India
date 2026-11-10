"""Civic Analyst: a tool-calling agent over the opinion database, ML outputs and the Wikipedia knowledge base.

The model plans its own sequence of tool calls (function calling), observes the results and iterates until it
can answer with citations. Every step is recorded so the UI can show the full reasoning trace.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from factprotest import analytics, prompts
from factprotest.config import MODELS
from factprotest.data import parse_analysis
from factprotest.llm import LLMClient, LLMError
from factprotest.rag.retriever import HybridRetriever

MAX_STEPS = 8
MAX_RESULT_CHARS = 3000


@dataclass
class OpinionIndex:
    ids: list[str]
    embeddings: np.ndarray
    encode_query: Callable[[str], np.ndarray]

    def search(self, query: str, candidates: list[int], k: int) -> list[int]:
        if not candidates:
            return []
        sims = self.embeddings[candidates] @ self.encode_query(query)
        return [candidates[i] for i in np.argsort(-sims)[:k]]


@dataclass
class AnalystContext:
    protests: list[dict[str, Any]]
    opinions: list[dict[str, Any]]
    replies: list[dict[str, Any]]
    retriever: HybridRetriever
    opinion_index: OpinionIndex
    topics: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    coordinated: list[dict[str, Any]] = field(default_factory=list)

    def protest(self, protest_id: int) -> dict[str, Any]:
        for p in self.protests:
            if int(p["id"]) == int(protest_id):
                return p
        raise ValueError(f"unknown protest_id {protest_id}; call list_protests first")

    def opinion_rows(self, protest_id: int) -> list[int]:
        return [i for i, o in enumerate(self.opinions) if str(o["issue_id"]) == str(protest_id)]


def _ref(opinion: dict[str, Any]) -> str:
    return f"O:{str(opinion['id'])[:8]}"


def _refuted(opinion: dict[str, Any]) -> int:
    return sum(c.get("verdict") == "REFUTED" for c in parse_analysis(opinion).get("claims", []))


def tool_list_protests(ctx: AnalystContext) -> list[dict[str, Any]]:
    counts = Counter(str(o["issue_id"]) for o in ctx.opinions)
    return [{"protest_id": p["id"], "title": p["title"], "status": p.get("status"), "category": p.get("category"),
             "opinions": counts.get(str(p["id"]), 0)} for p in ctx.protests]


def tool_get_protest_stats(ctx: AnalystContext, protest_id: int) -> dict[str, Any]:
    protest = ctx.protest(protest_id)
    rows = [ctx.opinions[i] for i in ctx.opinion_rows(protest_id)]
    if not rows:
        return {"protest": protest["title"], "opinions": 0}
    summary = analytics.protest_summary(rows)
    states = Counter(o.get("state") for o in rows if o.get("state") not in (None, "", "Not Specified"))
    scored = [o for o in rows if parse_analysis(o).get("schema_version") == 2]
    return {
        "protest": protest["title"], "status": protest.get("status"), "opinions": summary["n"],
        "stance_counts": summary["counts"],
        "support_share_with_95ci": summary["support_share"],
        "polarization_leik_d_with_95ci": summary["polarization"],
        "mean_credibility_with_95ci": summary["mean_credibility"],
        "opinions_with_refuted_claim": sum(_refuted(o) > 0 for o in scored),
        "fact_checked_opinions": len(scored),
        "top_states": states.most_common(5),
        "coordinated_clusters": [
            {"posts": len(c["member_ids"]), "mean_cosine": c["mean_similarity"], "span_minutes": c["span_minutes"],
             "accounts": c["distinct_authors"], "stance": c["dominant_stance"],
             "example_refs": [f"O:{m[:8]}" for m in c["member_ids"][:3]]}
            for c in ctx.coordinated if int(c["protest_id"]) == int(protest_id)],
        "authority_replies": sum(r.get("issue_title") == protest["title"] for r in ctx.replies),
    }


def tool_search_opinions(ctx: AnalystContext, protest_id: int, query: str, stance: str | None = None,
                         k: int = 5, min_credibility: int | None = None) -> list[dict[str, Any]]:
    candidates = [i for i in ctx.opinion_rows(protest_id)
                  if (stance in (None, "", "Any") or ctx.opinions[i]["stance"] == stance)
                  and (min_credibility is None or (ctx.opinions[i].get("credibility_score") or 0) >= min_credibility)]
    hits = ctx.opinion_index.search(query, candidates, min(int(k), 8))
    return [{"ref": _ref(ctx.opinions[i]), "stance": ctx.opinions[i]["stance"], "state": ctx.opinions[i].get("state"),
             "credibility": ctx.opinions[i].get("credibility_score"), "refuted_claims": _refuted(ctx.opinions[i]),
             "text": ctx.opinions[i]["opinion_text"][:260]} for i in hits]


def tool_search_wikipedia(ctx: AnalystContext, query: str, protest_id: int | None = None, k: int = 4) -> list[dict[str, Any]]:
    hits = ctx.retriever.search(query, k=min(int(k), 6), protest_id=protest_id)
    return [{"ref": f"W:{h.chunk.chunk_id}", "source": h.chunk.citation, "url": h.chunk.url,
             "text": " ".join(h.chunk.text.split()[:110])} for h in hits]


def tool_get_topics(ctx: AnalystContext, protest_id: int) -> list[dict[str, Any]]:
    ctx.protest(protest_id)
    return ctx.topics.get(str(protest_id), [])


def tool_compare_states(ctx: AnalystContext, protest_id: int) -> dict[str, Any]:
    rows = [ctx.opinions[i] for i in ctx.opinion_rows(protest_id)]
    test = analytics.stance_by_state_test([o.get("state") or "Not Specified" for o in rows], [o["stance"] for o in rows], min_count=5)
    by_state: dict[str, list[str]] = {}
    for o in rows:
        by_state.setdefault(o.get("state") or "Not Specified", []).append(o["stance"])
    shares = {s: {"n": len(v), "support_share": round(analytics.support_share(v), 3)}
              for s, v in sorted(by_state.items(), key=lambda kv: -len(kv[1])) if len(v) >= 5}
    test.pop("table", None)
    return {"chi_square_test": test, "states_with_5plus_opinions": shares}


def tool_get_authority_replies(ctx: AnalystContext, protest_id: int) -> list[dict[str, Any]]:
    title = ctx.protest(protest_id)["title"]
    return [{"authority": r.get("authority_name"), "status": r.get("status"), "date": str(r.get("created_at", ""))[:10],
             "text": r.get("reply_text", "")[:600]} for r in ctx.replies if r.get("issue_title") == title]


def _fn(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties, "required": required}}}


PID = {"type": "integer", "description": "Protest id from list_protests"}
TOOL_SPECS = [
    _fn("list_protests", "List all protests with id, title, status and number of opinions.", {}, []),
    _fn("get_protest_stats", "Stance counts, support share and polarization (Leik's D) with bootstrap 95% CIs, mean "
        "credibility, fact-check results, top states and coordinated-campaign alerts for one protest.", {"protest_id": PID}, ["protest_id"]),
    _fn("search_opinions", "Semantic search over citizen opinions for one protest. Returns ref ids to cite.",
        {"protest_id": PID, "query": {"type": "string"},
         "stance": {"type": "string", "enum": ["Support", "Oppose", "Neutral", "Any"]},
         "k": {"type": "integer", "minimum": 1, "maximum": 8},
         "min_credibility": {"type": "integer", "minimum": 0, "maximum": 100}}, ["protest_id", "query"]),
    _fn("search_wikipedia", "Hybrid (BM25 + dense) search over the Wikipedia knowledge base for background facts.",
        {"query": {"type": "string"}, "protest_id": PID, "k": {"type": "integer", "minimum": 1, "maximum": 6}}, ["query"]),
    _fn("get_topics", "Machine-discovered discussion themes (embedding clusters) with keywords and stance mix.",
        {"protest_id": PID}, ["protest_id"]),
    _fn("compare_states", "Chi-square test of whether stance depends on state, plus per-state support share.",
        {"protest_id": PID}, ["protest_id"]),
    _fn("get_authority_replies", "Official replies from authorities for one protest.", {"protest_id": PID}, ["protest_id"]),
]
TOOLS: dict[str, Callable[..., Any]] = {
    "list_protests": tool_list_protests, "get_protest_stats": tool_get_protest_stats,
    "search_opinions": tool_search_opinions, "search_wikipedia": tool_search_wikipedia,
    "get_topics": tool_get_topics, "compare_states": tool_compare_states,
    "get_authority_replies": tool_get_authority_replies,
}


@dataclass
class AgentStep:
    tool: str
    arguments: dict[str, Any]
    result: Any
    latency_ms: float
    error: str | None = None


@dataclass
class AgentRun:
    question: str
    answer: str
    steps: list[AgentStep]
    llm_calls: int
    tokens: int
    seconds: float


def execute_tool(ctx: AnalystContext, name: str, raw_arguments: str) -> tuple[dict[str, Any], Any, str | None]:
    try:
        arguments = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return {}, None, f"arguments were not valid JSON: {raw_arguments[:200]}"
    if name not in TOOLS:
        return arguments, None, f"unknown tool {name}"
    try:
        return arguments, TOOLS[name](ctx, **arguments), None
    except (TypeError, ValueError, KeyError) as exc:
        return arguments, None, str(exc)


def run_agent(llm: LLMClient, ctx: AnalystContext, question: str, model: str = MODELS.agent,
              max_steps: int = MAX_STEPS, on_step: Callable[[AgentStep], None] | None = None) -> AgentRun:
    start = time.perf_counter()
    first_call = len(llm.records)
    messages: list[dict[str, Any]] = [{"role": "system", "content": prompts.AGENT_SYSTEM},
                                      {"role": "user", "content": question}]
    steps: list[AgentStep] = []
    answer = ""
    for _ in range(max_steps):
        message = llm.tool_step(messages=messages, tools=TOOL_SPECS, model=model, purpose="agent")
        if not message.tool_calls:
            answer = (message.content or "").strip()
            break
        messages.append({"role": "assistant", "content": message.content or "", "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in message.tool_calls]})
        for call in message.tool_calls:
            t0 = time.perf_counter()
            arguments, result, error = execute_tool(ctx, call.function.name, call.function.arguments)
            step = AgentStep(call.function.name, arguments, result, round((time.perf_counter() - t0) * 1000, 1), error)
            steps.append(step)
            if on_step:
                on_step(step)
            payload = json.dumps({"error": error} if error else result, ensure_ascii=False, default=str)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": payload[:MAX_RESULT_CHARS]})
    if not answer:
        messages.append({"role": "user", "content": "Step limit reached. Write the final answer now from the tool results above."})
        try:
            answer = llm.tool_step(messages=messages, tools=TOOL_SPECS, model=model, purpose="agent-final").content or ""
        except LLMError as exc:
            answer = f"The analyst could not finish: {exc}"
    calls = list(llm.records)[first_call:]
    return AgentRun(question, answer.strip(), steps, len(calls), sum(c.total_tokens for c in calls),
                    round(time.perf_counter() - start, 1))


CITATION_RE = re.compile(r"[\[【]\s*(W:[^\]】]+)[\]】]")


def link_citations(answer: str, ctx: AnalystContext) -> tuple[str, list[tuple[int, str, str]]]:
    """Replace [W:chunk_id] (or [W:a, W:b]) markers with numbered links and return the source list."""
    by_id = {c.chunk_id: c for c in ctx.retriever.chunks}
    numbering: dict[str, int] = {}
    sources: list[tuple[int, str, str]] = []

    def replace(match: re.Match[str]) -> str:
        links = []
        for ref in re.split(r"[,;\s]+", match.group(1)):
            chunk = by_id.get(ref.removeprefix("W:").strip())
            if chunk is None:
                continue
            if chunk.citation not in numbering:
                numbering[chunk.citation] = len(numbering) + 1
                sources.append((numbering[chunk.citation], chunk.citation, chunk.url))
            link = f"[[{numbering[chunk.citation]}]]({chunk.url})"
            if link not in links:
                links.append(link)
        return "".join(links)

    return CITATION_RE.sub(replace, answer), sources
