"""Claim-level fact-checking pipeline.

    opinion ─► [1] analyse (claims + rubric + stance) ─► [2] hybrid retrieval per claim
            ─► [3] evidence-grounded verification ─► [4] explainable credibility score

Stages 1 and 3 are batched so one LLM call serves several opinions/claims; the live app runs the same code
with a batch of one.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from factprotest import prompts
from factprotest.config import MODELS
from factprotest.llm import LLMClient
from factprotest.rag.retriever import Hit, HybridRetriever
from factprotest.schemas import (
    BatchAnalysis,
    BatchVerdicts,
    CheckedClaim,
    ClaimVerdict,
    EvidenceRef,
    OpinionAnalysis,
    OpinionAssessment,
    SafetyReport,
)
from factprotest.scoring import credibility

ANALYZE_BATCH = 5
VERIFY_BATCH = 8
EVIDENCE_K = 3
PASSAGE_WORDS = 110


@dataclass
class OpinionInput:
    id: str
    protest_id: int
    protest_title: str
    stance: str
    text: str
    safety: SafetyReport = field(default_factory=SafetyReport)


@dataclass
class ClaimTask:
    claim_id: str
    claim: str
    hits: list[Hit]


def _clean(text: str) -> str:
    return text.replace("<", "‹").replace(">", "›")


def _passage(hit: Hit) -> str:
    return " ".join(hit.chunk.text.split()[:PASSAGE_WORDS])


def analyze(llm: LLMClient, items: list[OpinionInput], model: str = MODELS.analyze) -> dict[str, OpinionAnalysis]:
    results: dict[str, OpinionAnalysis] = {}
    for start in range(0, len(items), ANALYZE_BATCH):
        batch = items[start:start + ANALYZE_BATCH]
        user = "\n\n".join(
            f'<opinion id="{it.id}" protest="{it.protest_title}" '
            f'demand="{prompts.protest_demand(it.protest_id, it.protest_title)}">\n{_clean(it.text)}\n</opinion>'
            for it in batch)
        out = llm.json(system=prompts.ANALYZE_SYSTEM, user=user, schema=BatchAnalysis, model=model,
                       purpose="analyze", max_tokens=900 * len(batch))
        results.update({r.opinion_id: r for r in out.results if r.opinion_id in {it.id for it in batch}})
    return results


def verify(llm: LLMClient, tasks: list[ClaimTask], model: str = MODELS.verify) -> dict[str, ClaimVerdict]:
    verdicts: dict[str, ClaimVerdict] = {}
    for start in range(0, len(tasks), VERIFY_BATCH):
        batch = tasks[start:start + VERIFY_BATCH]
        blocks = []
        for task in batch:
            evidence = "\n".join(
                f"[{task.claim_id}.E{j + 1}] ({hit.chunk.citation}) {_passage(hit)}" for j, hit in enumerate(task.hits)
            ) or "(no evidence retrieved)"
            blocks.append(f"### Claim {task.claim_id}: {_clean(task.claim)}\nEvidence:\n{evidence}")
        out = llm.json(system=prompts.VERIFY_SYSTEM, user="\n\n".join(blocks), schema=BatchVerdicts, model=model,
                       purpose="verify", max_tokens=700 * len(batch))
        verdicts.update({v.claim_id: v for v in out.verdicts if v.claim_id in {t.claim_id for t in batch}})
    return verdicts


def _evidence_refs(task: ClaimTask, verdict: ClaimVerdict | None) -> list[EvidenceRef]:
    cited = {e.split(".")[-1] for e in (verdict.evidence_ids if verdict else [])}
    hits = [h for j, h in enumerate(task.hits) if f"E{j + 1}" in cited] or task.hits[:1]
    return [EvidenceRef(chunk_id=h.chunk.chunk_id, title=h.chunk.citation, url=h.chunk.url,
                        snippet=_passage(h), score=round(h.score, 4)) for h in hits]


def stances_conflict(declared: str, predicted: str) -> bool:
    return {declared, predicted} == {"Support", "Oppose"}


def assess(llm: LLMClient, retriever: HybridRetriever, items: list[OpinionInput],
           analyze_model: str = MODELS.analyze, verify_model: str = MODELS.verify) -> dict[str, OpinionAssessment]:
    start = time.perf_counter()
    analyses = analyze(llm, items, analyze_model)
    missing = [it for it in items if it.id not in analyses]
    if missing:
        analyses.update(analyze(llm, missing, analyze_model))

    tasks: dict[str, list[ClaimTask]] = {}
    queries, owners = [], []
    for it in items:
        analysis = analyses.get(it.id)
        for claim in analysis.claims if analysis else []:
            queries.append(claim)
            owners.append((it, f"{len(owners) + 1}"))
    hits = retriever.search_many(queries, k=EVIDENCE_K, protest_ids=[it.protest_id for it, _ in owners])
    for (it, claim_id), claim, claim_hits in zip(owners, queries, hits):
        tasks.setdefault(it.id, []).append(ClaimTask(f"C{claim_id}", claim, claim_hits))
    verdicts = verify(llm, [t for ts in tasks.values() for t in ts], verify_model) if queries else {}

    elapsed = (time.perf_counter() - start) * 1000 / max(len(items), 1)
    assessments: dict[str, OpinionAssessment] = {}
    for it in items:
        analysis = analyses.get(it.id)
        if analysis is None:
            continue
        checked, claim_verdicts = [], []
        for task in tasks.get(it.id, []):
            verdict = verdicts.get(task.claim_id) or ClaimVerdict(
                claim_id=task.claim_id, verdict="NOT_ENOUGH_INFO", confidence=0.0, explanation="Verifier returned no verdict.")
            claim_verdicts.append(verdict)
            checked.append(CheckedClaim(claim=task.claim, verdict=verdict.verdict, confidence=verdict.confidence,
                                        explanation=verdict.explanation, evidence=_evidence_refs(task, verdict)))
        score = credibility(claim_verdicts, analysis.reasoning_quality, analysis.specificity, analysis.civility,
                            analysis.emotional_intensity, injection_flagged=it.safety.injection_flagged)
        if stances_conflict(it.stance, analysis.predicted_stance):
            score.flags.append("stance_mismatch")
        if it.safety.moderation_flagged:
            score.flags.append("moderation_flagged")
        assessments[it.id] = OpinionAssessment(
            claims=checked, reasoning_quality=analysis.reasoning_quality, specificity=analysis.specificity,
            civility=analysis.civility, emotional_intensity=analysis.emotional_intensity,
            predicted_stance=analysis.predicted_stance,
            stance_consistent=not stances_conflict(it.stance, analysis.predicted_stance),
            rationale=analysis.rationale, score=score, safety=it.safety,
            models={"analyze": analyze_model, "verify": verify_model, "embedding": MODELS.embedding},
            prompt_versions={"analyze": prompts.ANALYZE_VERSION, "verify": prompts.VERIFY_VERSION},
            latency_ms=round(elapsed, 1),
        )
    return assessments
