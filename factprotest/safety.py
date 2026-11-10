"""Input safety: prompt-injection detection (Llama Prompt Guard 2) and policy moderation (gpt-oss-safeguard)."""
from __future__ import annotations

from pydantic import BaseModel, Field

from factprotest import prompts
from factprotest.config import MODELS
from factprotest.llm import LLMClient, LLMError
from factprotest.schemas import SafetyReport

INJECTION_THRESHOLD = 0.5


class ModerationResult(BaseModel):
    violation: int = Field(ge=0, le=1)
    category: str = "none"
    rationale: str = ""


def injection_score(llm: LLMClient, text: str) -> float | None:
    try:
        raw = llm.classify(model=MODELS.prompt_guard, text=text[:2000], purpose="prompt_guard")
        return float(raw)
    except (LLMError, ValueError):
        return None


def moderate(llm: LLMClient, text: str) -> ModerationResult | None:
    try:
        return llm.json(system=prompts.MODERATION_POLICY, user=text[:4000], schema=ModerationResult,
                        model=MODELS.safeguard, purpose="moderation", max_tokens=600)
    except LLMError:
        return None


def check(llm: LLMClient, text: str, moderation: bool = True) -> SafetyReport:
    score = injection_score(llm, text)
    report = SafetyReport(injection_score=score, injection_flagged=score is not None and score >= INJECTION_THRESHOLD)
    if moderation:
        result = moderate(llm, text)
        if result is not None:
            report.moderation_flagged = bool(result.violation)
            report.moderation_category = None if result.category == "none" else result.category
    return report
