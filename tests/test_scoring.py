import pytest

from factprotest.schemas import ClaimVerdict
from factprotest.scoring import INJECTION_CAP, credibility, evidence_component


def v(verdict: str, confidence: float = 1.0) -> ClaimVerdict:
    return ClaimVerdict(claim_id="c", verdict=verdict, confidence=confidence)


def test_evidence_component_maps_verdicts_to_0_100():
    assert evidence_component([v("SUPPORTED")]) == 100
    assert evidence_component([v("REFUTED")]) == 0
    assert evidence_component([v("NOT_ENOUGH_INFO")]) == 50
    assert evidence_component([v("SUPPORTED"), v("REFUTED")]) == 50
    assert evidence_component([]) is None


def test_no_claims_reweights_reasoning_and_civility():
    score = credibility([], reasoning_quality=5, specificity=5, civility=5, emotional_intensity=1)
    assert score.evidence is None
    assert score.weights == pytest.approx({"reasoning": 0.667, "civility": 0.333}, abs=1e-3)
    assert score.final == 100
    assert "no_checkable_claims" in score.flags


def test_refuted_claim_is_flagged_and_lowers_score():
    good = credibility([v("SUPPORTED", 0.9)], 4, 4, 5, 2)
    bad = credibility([v("REFUTED", 0.9)], 4, 4, 5, 2)
    assert bad.final < good.final
    assert "contains_refuted_claim" in bad.flags


def test_prompt_injection_caps_score():
    score = credibility([v("SUPPORTED")], 5, 5, 5, 1, injection_flagged=True)
    assert score.final == INJECTION_CAP
    assert "prompt_injection_detected" in score.flags


def test_verdict_aliases_are_normalised():
    assert ClaimVerdict(claim_id="c", verdict="not enough info", confidence=0.5).verdict == "NOT_ENOUGH_INFO"
    assert ClaimVerdict(claim_id="c", verdict="Refutes", confidence=0.5).verdict == "REFUTED"
