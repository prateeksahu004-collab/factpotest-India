from __future__ import annotations

import uuid
from collections import Counter
from typing import Any

import streamlit as st
from pydantic import BaseModel, Field

from factprotest.config import MODELS
from factprotest.data import parse_analysis
from factprotest.llm import LLMError
from factprotest.scoring import credibility_label
from ui import resources
from ui.components import render_assessment, run_factcheck
from ui.style import CATEGORY_COLOR, STANCE_COLOR, badge, esc, kpis, section, stance_split

INDIAN_STATES = ["Not Specified", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
                 "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
                 "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
                 "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands", "Chandigarh",
                 "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"]
STATUS_COLOR = {"Ongoing": "#d97706", "Authority Replied": "#2563eb", "Resolved": "#16a34a"}


class ReplyRating(BaseModel):
    specificity: int = Field(ge=1, le=10)
    addresses_concerns: int = Field(ge=1, le=10)
    commits_to_action: int = Field(ge=1, le=10)
    overall: int = Field(ge=1, le=10)
    assessment: str


def _by_protest(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for o in rows:
        grouped.setdefault(str(o["issue_id"]), []).append(o)
    return grouped


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [o["credibility_score"] for o in rows if o.get("credibility_score") is not None]
    checked = [parse_analysis(o) for o in rows]
    checked = [a for a in checked if a.get("schema_version") == 2]
    return {"n": len(rows), "counts": Counter(o["stance"] for o in rows),
            "avg": round(sum(scores) / len(scores)) if scores else None,
            "checked": len(checked),
            "refuted": sum(any(c["verdict"] == "REFUTED" for c in a["claims"]) for a in checked)}


def page_home() -> None:
    st.html('<div class="hero"><h1>🇮🇳 FactProtest India</h1><p>Claim-level fact-checking, explainable credibility '
            'scores and AI-agent analysis of public opinion on Indian protests.</p>'
            '<span class="pill">RAG over Wikipedia</span><span class="pill">Hybrid BM25 + dense retrieval</span>'
            '<span class="pill">Tool-calling AI agent</span><span class="pill">Stance & topic ML</span>'
            '<span class="pill">Coordinated-campaign detection</span><span class="pill">Evaluated, not vibes</span></div>')
    protests = resources.protests()
    if not protests:
        st.warning("No protests found. Check the database connection.")
        return
    grouped = _by_protest(resources.opinions())
    c1, c2, c3 = st.columns([3, 1.5, 1.5])
    search = c1.text_input("Search protests", placeholder="e.g. NEET, farm laws, Ladakh")
    level = c2.selectbox("Level", ["All", "Central", "State"])
    status = c3.selectbox("Status", ["All", "Ongoing", "Authority Replied", "Resolved"])
    shown = [p for p in protests
             if (not search or search.lower() in (p["title"] + p["description"]).lower())
             and level in ("All", p.get("level")) and status in ("All", p.get("status"))]
    st.caption(f"{len(shown)} protest(s)")
    for p in shown:
        s = _stats(grouped.get(str(p["id"]), []))
        score_badge = badge(f"avg credibility {s['avg']}", credibility_label(s["avg"])[1]) if s["avg"] is not None else ""
        refuted = f" · ⚠️ {s['refuted']} with refuted claims" if s["checked"] else ""
        st.html(f'<div class="card"><h3>{esc(p["title"])}</h3>'
                f'{badge(p.get("category", ""), CATEGORY_COLOR.get(p.get("category"), "#475569"))}'
                f'{badge(p.get("status", "Ongoing"), STATUS_COLOR.get(p.get("status"), "#475569"))}{score_badge}'
                f'<div class="muted" style="margin:6px 0">📍 {esc(p.get("location", ""))} · 📅 {esc(p.get("date_started", ""))} · '
                f'🏛 {esc(p.get("level", ""))}</div><div style="font-size:0.86rem;color:#334155">{esc(p["description"][:220])}…</div>'
                f'<div class="muted" style="margin-top:6px">💬 {s["n"]} opinions · {s["checked"]} fact-checked{refuted}</div>'
                + (stance_split(dict(s["counts"])) if s["n"] else "") + "</div>")
        if st.button("Open →", key=f"open_{p['id']}"):
            st.session_state["page"], st.session_state["sel"] = "detail", p["id"]
            st.rerun()


def _rate_reply(protest_title: str, reply_text: str) -> ReplyRating | None:
    try:
        return resources.llm().json(
            system="Rate an official government reply to a protest on 1-10 scales: specificity, addresses_concerns, "
                   "commits_to_action, overall; plus a one-sentence assessment. Return JSON with exactly those keys.",
            user=f"Protest: {protest_title}\nReply:\n{reply_text}", schema=ReplyRating, model=MODELS.analyze,
            purpose="rate_reply", max_tokens=800)
    except LLMError as exc:
        st.error(f"Rating failed: {exc}")
        return None


def page_detail() -> None:
    protest = next((p for p in resources.protests() if p["id"] == st.session_state.get("sel")), None)
    if st.button("← All protests"):
        st.session_state["page"] = "home"
        st.rerun()
    if not protest:
        st.error("Protest not found.")
        return
    rows = _by_protest(resources.opinions()).get(str(protest["id"]), [])
    s = _stats(rows)
    st.html(f'<div class="hero"><div>{badge(protest.get("category", ""), CATEGORY_COLOR.get(protest.get("category"), "#475569"))}'
            f'{badge(protest.get("status", ""), STATUS_COLOR.get(protest.get("status"), "#475569"))}</div>'
            f'<h1 style="font-size:1.45rem;margin-top:8px">{esc(protest["title"])}</h1><p>{esc(protest["description"])}</p>'
            f'<p style="margin-top:6px;font-size:0.8rem">📍 {esc(protest.get("location", ""))} · 📅 {esc(protest.get("date_started", ""))}</p></div>')
    kpis([("Opinions", str(s["n"]), f'{s["checked"]} fact-checked'),
          ("Support", str(s["counts"].get("Support", 0)), ""), ("Oppose", str(s["counts"].get("Oppose", 0)), ""),
          ("Avg credibility", "–" if s["avg"] is None else str(s["avg"]), "0-100, explainable"),
          ("Refuted-claim opinions", str(s["refuted"]), "checked against Wikipedia")])
    if s["n"]:
        st.html(stance_split(dict(s["counts"])))

    if protest.get("status") != "Resolved":
        section("📝 Share your opinion. It will be fact-checked claim by claim")
        with st.form(f"form_{protest['id']}", clear_on_submit=False):
            name = st.text_input("Your name *")
            state = st.selectbox("Your state", INDIAN_STATES)
            stance = st.radio("Stance", ["Support", "Oppose", "Neutral"], horizontal=True)
            text = st.text_area("Your opinion *", height=120, placeholder="State your view and the facts behind it…")
            submitted = st.form_submit_button("🚀 Submit & fact-check", use_container_width=True)
        if submitted:
            if not name.strip() or len(text.strip()) < 15:
                st.error("Please add your name and an opinion of at least 15 characters.")
            else:
                opinion_id = str(uuid.uuid4())
                result = run_factcheck(opinion_id, protest, stance, text.strip())
                if result is None:
                    st.error("The AI pipeline is busy (free-tier rate limit). Please try again in a minute.")
                else:
                    resources.repo().add_opinion({
                        "issue_id": str(protest["id"]), "issue_title": protest["title"], "user_name": name.strip(),
                        "state": state, "stance": stance, "opinion_text": text.strip(),
                        "credibility_score": result.score.final, "ai_analysis": result.model_dump_json()})
                    resources.refresh()
                    st.success("Saved. Here is how your opinion was assessed:")
                    render_assessment(result.model_dump(), stance, text.strip())

    topics = resources.topics().get(str(protest["id"]), [])
    if topics:
        section("🧭 What people are talking about (ML-discovered themes)")
        cols = st.columns(min(3, len(topics)))
        for col, t in zip(cols * 3, topics):
            mix = " ".join(badge(f"{k} {v}", STANCE_COLOR.get(k, "#475569")) for k, v in t["stance_mix"].items())
            col.html(f'<div class="card"><h3>{esc(t["label"])}</h3><div class="muted">{t["size"]} opinions · '
                     f'{esc(", ".join(t["keywords"][:5]))}</div><div style="font-size:0.82rem;margin:6px 0">{esc(t["summary"])}</div>{mix}</div>')

    section("🏛 Authority response")
    protest_replies = [r for r in resources.replies() if r.get("issue_title") == protest["title"]]
    if not protest_replies:
        st.info("No authority reply yet.")
    for rep in protest_replies:
        st.html(f'<div class="reply"><b>🏛 {esc(rep.get("authority_name", "Authority"))}</b> '
                f'<span class="muted">· {esc(rep.get("status", ""))} · {esc(str(rep.get("created_at", ""))[:10])}</span>'
                f'<div style="font-size:0.88rem;margin-top:6px">{esc(rep.get("reply_text", ""))}</div></div>')
        up, down = resources.repo().reactions(protest["id"])
        c1, c2, c3 = st.columns([1, 1, 3])
        if c1.button(f"👍 {up}", key=f"up_{rep['id']}"):
            resources.repo().add_reaction(protest["id"], "up")
            st.rerun()
        if c2.button(f"👎 {down}", key=f"dn_{rep['id']}"):
            resources.repo().add_reaction(protest["id"], "down")
            st.rerun()
        key = f"rating_{rep['id']}"
        if c3.button("🤖 Rate this reply's quality", key=f"rate_{rep['id']}"):
            with st.spinner("Rating…"):
                st.session_state[key] = _rate_reply(protest["title"], rep.get("reply_text", ""))
        if st.session_state.get(key):
            r: ReplyRating = st.session_state[key]
            kpis([("Specificity", f"{r.specificity}/10", ""), ("Addresses concerns", f"{r.addresses_concerns}/10", ""),
                  ("Commits to action", f"{r.commits_to_action}/10", ""), ("Overall", f"{r.overall}/10", "")])
            st.caption(r.assessment)
    if protest.get("status") == "Resolved" and protest.get("outcome"):
        st.success(f"Outcome: {protest['outcome']}")

    section("💬 Recent opinions")
    if rows and all(parse_analysis(o).get("source") == "synthetic" for o in rows[:20]):
        st.html('<div class="synthetic">Demo data: these opinions are synthetic (LLM-generated from persona specs) and '
                'scored by the production pipeline (scoring is rate-limited, so some are still queued).</div>')
    stance_filter = st.radio("Show", ["All", "Support", "Oppose", "Neutral", "Has refuted claim"], horizontal=True)
    filtered = [o for o in rows if stance_filter in ("All", o["stance"]) or (
        stance_filter == "Has refuted claim" and any(c["verdict"] == "REFUTED" for c in parse_analysis(o).get("claims", [])))]
    for o in filtered[:12]:
        analysis = parse_analysis(o)
        score = o.get("credibility_score")
        score_badge = (badge(f"credibility {score}", credibility_label(score)[1]) if score is not None
                       else badge("not yet fact-checked", "#94a3b8"))
        st.html(f'<div class="card">{badge(o["stance"], STANCE_COLOR.get(o["stance"], "#475569"))}'
                f'<b>{esc(o.get("user_name", "Anonymous"))}</b> <span class="muted">· {esc(o.get("state", ""))} · '
                f'{esc(str(o.get("created_at", ""))[:10])}</span><span style="float:right">{score_badge}</span>'
                f'<div style="font-size:0.87rem;color:#334155;margin-top:6px">{esc(o["opinion_text"])}</div></div>')
        if analysis.get("schema_version") == 2:
            with st.expander("Why this score?"):
                render_assessment(analysis, o["stance"])
