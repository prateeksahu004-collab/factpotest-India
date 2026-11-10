from __future__ import annotations

import hmac
from collections import Counter

import plotly.graph_objects as go
import streamlit as st

from factprotest.config import secret
from factprotest.data import parse_analysis
from ui import resources
from ui.style import STANCE_COLOR, esc, kpis, section


def page_replies() -> None:
    st.title("🏛 Authority Replies")
    replies = resources.replies()
    if not replies:
        st.info("No authority replies yet.")
    for title in dict.fromkeys(r["issue_title"] for r in replies):
        st.markdown(f"### {title}")
        for rep in (r for r in replies if r["issue_title"] == title):
            st.html(f'<div class="reply"><b>🏛 {esc(rep.get("authority_name"))}</b> <span class="muted">· '
                    f'{esc(rep.get("status"))} · {esc(str(rep.get("created_at", ""))[:10])}</span>'
                    f'<div style="font-size:0.88rem;margin-top:6px">{esc(rep.get("reply_text"))}</div></div>')


def page_authority() -> None:
    st.title("🏛 Authority Portal")
    if not st.session_state.get("auth_in"):
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            if hmac.compare_digest(pwd, secret("AUTHORITY_PWD")):
                st.session_state["auth_in"] = True
                st.rerun()
            st.error("Wrong password.")
        return
    if st.button("Logout"):
        st.session_state["auth_in"] = False
        st.rerun()
    protests = resources.protests()
    rows = resources.opinions()
    analyses = {str(o["id"]): parse_analysis(o) for o in rows}
    checked = [a for a in analyses.values() if a.get("schema_version") == 2]
    kpis([("Opinions", str(len(rows)), f"{len(checked)} fact-checked"),
          ("Refuted claims", str(sum(c["verdict"] == "REFUTED" for a in checked for c in a["claims"])), "across all opinions"),
          ("Safety flags", str(sum(a["safety"].get("injection_flagged") or a["safety"].get("moderation_flagged") for a in checked)),
           "injection or policy"),
          ("Most active state", Counter(o.get("state") for o in rows if o.get("state") not in (None, "Not Specified")).most_common(1)[0][0]
           if rows else "–", "")])

    section("Citizen sentiment by protest")
    fig = go.Figure()
    short = [p["title"].split("—")[0].strip()[:26] for p in protests]
    for stance in ("Support", "Neutral", "Oppose"):
        fig.add_bar(y=short, x=[sum(1 for o in rows if str(o["issue_id"]) == str(p["id"]) and o["stance"] == stance) for p in protests],
                    name=stance, orientation="h", marker_color=STANCE_COLOR[stance])
    fig.update_layout(barmode="stack", height=380, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    section("Review queue: opinions the pipeline flagged")
    flagged = [o for o in rows if analyses[str(o["id"])].get("schema_version") == 2
               and set(analyses[str(o["id"])]["score"]["flags"]) & {"contains_refuted_claim", "prompt_injection_detected", "moderation_flagged"}]
    st.caption(f"{len(flagged)} opinions contain a refuted claim or tripped a safety check.")
    for o in flagged[:15]:
        a = analyses[str(o["id"])]
        refuted = [c for c in a["claims"] if c["verdict"] == "REFUTED"]
        st.html(f'<div class="card"><b>{esc(o.get("user_name"))}</b> <span class="muted">on {esc(o["issue_title"][:50])} · '
                f'flags: {esc(", ".join(a["score"]["flags"]))}</span><div style="font-size:0.85rem;margin:6px 0">{esc(o["opinion_text"])}</div>'
                + "".join(f'<div class="evidence">❌ “{esc(c["claim"])}”: {esc(c["explanation"])}</div>' for c in refuted) + "</div>")

    section("Submit an official reply")
    pick = st.selectbox("Protest", [p["title"] for p in protests])
    protest = next(p for p in protests if p["title"] == pick)
    with st.form("reply_form", clear_on_submit=True):
        name = st.text_input("Ministry / authority name")
        text = st.text_area("Official reply", height=150)
        status = st.selectbox("Status", ["Acknowledged", "Under Review", "Action Taken", "Resolved"])
        if st.form_submit_button("Submit reply"):
            if name.strip() and text.strip():
                resources.repo().add_reply({"issue_id": str(protest["id"]), "issue_title": protest["title"],
                                            "authority_name": name.strip(), "reply_text": text.strip(), "status": status})
                resources.refresh()
                st.success("Reply published.")
            else:
                st.error("Name and reply are required.")
