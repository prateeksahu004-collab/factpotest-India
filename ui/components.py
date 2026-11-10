"""Reusable UI pieces: live fact-check runner and the explainable assessment view."""
from __future__ import annotations

from typing import Any

import streamlit as st

from factprotest import safety
from factprotest.factcheck import OpinionInput, assess
from factprotest.schemas import OpinionAssessment
from factprotest.scoring import credibility_label
from ui import resources
from ui.style import STANCE_COLOR, VERDICT_COLOR, badge, bar, esc

FLAG_TEXT = {
    "no_checkable_claims": ("No checkable facts", "#64748b"),
    "contains_refuted_claim": ("Contains a refuted claim", "#dc2626"),
    "prompt_injection_detected": ("Prompt-injection attempt", "#7c3aed"),
    "stance_mismatch": ("Text contradicts declared stance", "#d97706"),
    "moderation_flagged": ("Flagged by safety policy", "#be123c"),
}


def run_factcheck(opinion_id: str, protest: dict[str, Any], stance: str, text: str) -> OpinionAssessment | None:
    llm = resources.llm()
    item = OpinionInput(opinion_id, int(protest["id"]), protest["title"], stance, text)
    with st.status("Fact-checking your opinion…", expanded=True) as status:
        st.write("🛡️ Screening for prompt injection (Llama Prompt Guard 2) and policy violations (gpt-oss-safeguard)…")
        item.safety = safety.check(llm, text, moderation=True)
        st.write("🧩 Extracting check-worthy claims and rating the reasoning (gpt-oss-20b)…")
        st.write("🔎 Retrieving Wikipedia evidence (BM25 + bge-small dense, fused with RRF)…")
        st.write("⚖️ Verifying each claim against the evidence (gpt-oss-120b)…")
        result = assess(llm, resources.retriever(), [item]).get(opinion_id)
        status.update(label="Fact-check complete" if result else "Fact-check failed",
                      state="complete" if result else "error", expanded=False)
    return result


def ml_stance(text: str) -> dict[str, float] | None:
    model = resources.stance_model()
    if model is None:
        return None
    from factprotest.rag.embed import get_embedder

    return model.predict_proba([text], embed=lambda t: get_embedder().documents(t))[0]


def render_assessment(a: dict[str, Any], declared_stance: str | None = None, text: str | None = None) -> None:
    if a.get("schema_version") != 2:
        st.caption("Legacy v1 score: produced before claim-level fact-checking existed, so it cannot be explained.")
        return
    score = a["score"]
    label, color = credibility_label(score["final"])
    left, right = st.columns([1, 2])
    with left:
        st.html(f'<div class="card" style="text-align:center"><div class="muted">Credibility</div>'
                f'<div style="font-size:2.6rem;font-weight:800;color:{color}">{score["final"]}</div>'
                f'{badge(label, color)}</div>')
    with right:
        rows = []
        for key, name, col in (("evidence", "Evidence (claims vs Wikipedia)", "#2563eb"),
                               ("reasoning", "Reasoning quality & specificity", "#7c3aed"),
                               ("civility", "Civility & emotional restraint", "#0d9488")):
            weight = score["weights"].get(key)
            value = score.get(key)
            shown = "n/a (no factual claims)" if value is None else f"{value:.0f}/100 · weight {weight:.0%}"
            rows.append(f'<div style="font-size:0.8rem;color:#334155">{name}: <b>{shown}</b></div>{bar(value, col)}')
        flags = "".join(badge(*FLAG_TEXT.get(f, (f, "#334155"))) for f in score.get("flags", []))
        st.html("".join(rows) + (f'<div style="margin-top:4px">{flags}</div>' if flags else ""))

    claims = a.get("claims", [])
    if claims:
        st.markdown(f"**Claims checked ({len(claims)})**")
        for c in claims:
            evidence = "".join(
                f'<div class="evidence">📖 <a href="{esc(e["url"])}" target="_blank">{esc(e["title"])}</a><br>'
                f'{esc(e["snippet"][:420])}…</div>' for e in c.get("evidence", []))
            st.html(f'<div class="verdict {c["verdict"]}">{badge(c["verdict"].replace("_", " "), VERDICT_COLOR[c["verdict"]])}'
                    f'<span class="muted">confidence {c["confidence"]:.0%}</span>'
                    f'<div style="margin:6px 0 2px;font-weight:600;color:#0f172a">“{esc(c["claim"])}”</div>'
                    f'<div style="font-size:0.82rem;color:#475569">{esc(c["explanation"])}</div>{evidence}</div>')
    else:
        st.caption("No check-worthy factual claims found: this is a pure opinion, so it is scored on reasoning and civility only.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reasoning", f'{a["reasoning_quality"]}/5')
    c2.metric("Specificity", f'{a["specificity"]}/5')
    c3.metric("Civility", f'{a["civility"]}/5')
    c4.metric("Emotional intensity", f'{a["emotional_intensity"]}/5')
    if a.get("rationale"):
        st.caption(f"Rubric rationale: {a['rationale']}")

    stance_bits = []
    if declared_stance:
        stance_bits.append(f"declared {badge(declared_stance, STANCE_COLOR[declared_stance])}")
    stance_bits.append(f"LLM reads {badge(a['predicted_stance'], STANCE_COLOR[a['predicted_stance']])}")
    if text:
        probs = ml_stance(text)
        if probs:
            top = max(probs, key=probs.get)
            stance_bits.append(f"ML classifier {badge(f'{top} {probs[top]:.0%}', STANCE_COLOR[top])}")
    safety_info = a.get("safety", {})
    inj = safety_info.get("injection_score")
    safety_txt = "not run" if inj is None else f"injection risk {inj:.1%}"
    if safety_info.get("moderation_flagged"):
        safety_txt += f" · policy flag: {safety_info.get('moderation_category')}"
    st.html(f'<div class="muted" style="margin-top:6px">Stance: {" · ".join(stance_bits)}<br>Safety: {esc(safety_txt)}'
            f' · models: {esc(", ".join(f"{k}={v}" for k, v in a.get("models", {}).items()))}'
            f' · prompts: {esc(", ".join(a.get("prompt_versions", {}).values()))}'
            + (f' · {a["latency_ms"]:.0f} ms' if a.get("latency_ms") else "") + "</div>")
