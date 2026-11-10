"""Explainable credibility score: a weighted, documented combination of evidence, reasoning and civility.

Replaces the v1 approach of asking an LLM for a single opaque 0-100 number. Each component is reported
separately so a user can see *why* an opinion scored the way it did.
"""
from __future__ import annotations

from factprotest.schemas import ClaimVerdict, ScoreBreakdown

WEIGHTS = {"evidence": 0.40, "reasoning": 0.40, "civility": 0.20}
INJECTION_CAP = 20


def evidence_component(verdicts: list[ClaimVerdict]) -> float | None:
    """Mean confidence-weighted verdict in [-1, 1] mapped to 0-100. None when there are no factual claims."""
    if not verdicts:
        return None
    signed = []
    for v in verdicts:
        sign = {"SUPPORTED": 1.0, "REFUTED": -1.0}.get(v.verdict, 0.0)
        signed.append(sign * v.confidence)
    return round(50 * (1 + sum(signed) / len(signed)), 1)


def reasoning_component(reasoning_quality: int, specificity: int) -> float:
    return round(100 * (0.6 * (reasoning_quality - 1) / 4 + 0.4 * (specificity - 1) / 4), 1)


def civility_component(civility: int, emotional_intensity: int) -> float:
    return round(100 * (0.7 * (civility - 1) / 4 + 0.3 * (5 - emotional_intensity) / 4), 1)


def credibility(verdicts: list[ClaimVerdict], reasoning_quality: int, specificity: int, civility: int,
                emotional_intensity: int, injection_flagged: bool = False) -> ScoreBreakdown:
    evidence = evidence_component(verdicts)
    reasoning = reasoning_component(reasoning_quality, specificity)
    civil = civility_component(civility, emotional_intensity)
    weights = dict(WEIGHTS)
    flags: list[str] = []
    if evidence is None:
        weights.pop("evidence")
        flags.append("no_checkable_claims")
    total = sum(weights.values())
    weights = {k: round(v / total, 3) for k, v in weights.items()}
    parts = {"evidence": evidence, "reasoning": reasoning, "civility": civil}
    final = sum(weights[k] * parts[k] for k in weights)  # type: ignore[operator]
    if any(v.verdict == "REFUTED" and v.confidence >= 0.6 for v in verdicts):
        flags.append("contains_refuted_claim")
    if injection_flagged:
        final = min(final, INJECTION_CAP)
        flags.append("prompt_injection_detected")
    return ScoreBreakdown(evidence=evidence, reasoning=reasoning, civility=civil, final=int(round(final)),
                          weights=weights, flags=flags)


def credibility_label(score: float) -> tuple[str, str]:
    if score >= 75:
        return "Highly credible", "#15803d"
    if score >= 55:
        return "Moderately credible", "#ca8a04"
    if score >= 35:
        return "Low credibility", "#ea580c"
    return "Very low credibility", "#dc2626"
