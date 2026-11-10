"""Typed contracts for every LLM output and for the stored opinion assessment."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Stance = Literal["Support", "Oppose", "Neutral"]
Verdict = Literal["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
VERDICTS: tuple[Verdict, ...] = ("SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO")


def _normalize_verdict(value: str) -> str:
    v = str(value).strip().upper().replace(" ", "_").replace("-", "_")
    aliases = {"NEI": "NOT_ENOUGH_INFO", "NOT_ENOUGH_INFORMATION": "NOT_ENOUGH_INFO", "INSUFFICIENT": "NOT_ENOUGH_INFO",
               "SUPPORTS": "SUPPORTED", "TRUE": "SUPPORTED", "REFUTES": "REFUTED", "FALSE": "REFUTED"}
    return aliases.get(v, v)


def _normalize_stance(value: str) -> str:
    v = str(value).strip().capitalize()
    return {"Supports": "Support", "For": "Support", "Against": "Oppose", "Opposes": "Oppose", "Mixed": "Neutral"}.get(v, v)


class OpinionAnalysis(BaseModel):
    opinion_id: str
    claims: list[str] = Field(default_factory=list)
    reasoning_quality: int = Field(ge=1, le=5)
    specificity: int = Field(ge=1, le=5)
    civility: int = Field(ge=1, le=5)
    emotional_intensity: int = Field(ge=1, le=5)
    predicted_stance: Stance
    rationale: str = ""

    @field_validator("claims")
    @classmethod
    def _cap_claims(cls, claims: list[str]) -> list[str]:
        return [c.strip() for c in claims if c and len(c.strip()) > 8][:3]

    @field_validator("predicted_stance", mode="before")
    @classmethod
    def _stance(cls, v: str) -> str:
        return _normalize_stance(v)


class BatchAnalysis(BaseModel):
    results: list[OpinionAnalysis]


class ClaimVerdict(BaseModel):
    claim_id: str
    verdict: Verdict
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(default_factory=list)
    explanation: str = ""

    @field_validator("verdict", mode="before")
    @classmethod
    def _verdict(cls, v: str) -> str:
        return _normalize_verdict(v)

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def _ids(cls, v: object) -> list[str]:
        if v is None:
            return []
        return [str(x) for x in v] if isinstance(v, list) else [str(v)]


class BatchVerdicts(BaseModel):
    verdicts: list[ClaimVerdict]


class StancePrediction(BaseModel):
    id: str
    stance: Stance

    @field_validator("stance", mode="before")
    @classmethod
    def _stance(cls, v: str) -> str:
        return _normalize_stance(v)


class BatchStance(BaseModel):
    predictions: list[StancePrediction]


class EvidenceRef(BaseModel):
    chunk_id: str
    title: str
    url: str
    snippet: str
    score: float


class CheckedClaim(BaseModel):
    claim: str
    verdict: Verdict
    confidence: float
    explanation: str
    evidence: list[EvidenceRef]


class ScoreBreakdown(BaseModel):
    evidence: float | None
    reasoning: float
    civility: float
    final: int
    weights: dict[str, float]
    flags: list[str] = Field(default_factory=list)


class SafetyReport(BaseModel):
    injection_score: float | None = None
    injection_flagged: bool = False
    moderation_flagged: bool = False
    moderation_category: str | None = None


class OpinionAssessment(BaseModel):
    """Stored as JSON in opinions.ai_analysis (schema_version 2)."""

    schema_version: int = 2
    claims: list[CheckedClaim]
    reasoning_quality: int
    specificity: int
    civility: int
    emotional_intensity: int
    predicted_stance: Stance
    stance_consistent: bool
    rationale: str
    score: ScoreBreakdown
    safety: SafetyReport = Field(default_factory=SafetyReport)
    models: dict[str, str]
    prompt_versions: dict[str, str]
    latency_ms: float | None = None
