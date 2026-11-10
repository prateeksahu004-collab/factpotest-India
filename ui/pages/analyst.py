from __future__ import annotations

import json

import streamlit as st

from factprotest.agent import AgentStep, link_citations, run_agent
from factprotest.llm import LLMError
from ui import resources

EXAMPLE_QUESTIONS = [
    "Which protest is the most polarized, and what are the strongest arguments on each side?",
    "What do opponents of the Agnipath protest argue, and do their factual claims hold up against Wikipedia?",
    "Does stance on the farm-laws protest differ significantly by state?",
    "Are there signs of coordinated campaigns in any protest? Show evidence.",
]


def _render_step(step: AgentStep, container: st.delta_generator.DeltaGenerator) -> None:
    args = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in step.arguments.items())
    status = f"⚠️ {step.error}" if step.error else f"{step.latency_ms:.0f} ms"
    container.html(f'<div class="step">→ {step.tool}({args})  <span style="color:#94a3b8">[{status}]</span></div>')


def page_analyst() -> None:
    st.title("🤖 Civic Analyst Agent")
    st.caption("A tool-calling agent (gpt-oss-120b) that plans its own steps over 7 tools: database statistics with "
               "bootstrap confidence intervals, semantic opinion search, hybrid Wikipedia retrieval, ML topic clusters, "
               "chi-square state tests, coordination alerts and authority replies. Every answer is grounded in tool "
               "output and cited.")
    protests = resources.protests()
    tab_ask, tab_brief = st.tabs(["Ask a question", "Generate a civic brief"])
    with tab_ask:
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            if cols[i % 2].button(q, key=f"ex_{i}", use_container_width=True):
                st.session_state["agent_q"] = q
        question = st.text_area("Your question", key="agent_q", height=80)
        go = st.button("Run analyst", type="primary", disabled=not question.strip())
    with tab_brief:
        pick = st.selectbox("Protest", [p["title"] for p in protests])
        brief = st.button("Write civic brief", type="primary")
        if brief:
            pid = next(p["id"] for p in protests if p["title"] == pick)
            question = (f"Write a neutral civic brief on protest_id {pid} ({pick}) for a policymaker: background (cite "
                        "Wikipedia), public sentiment with uncertainty, main themes, the strongest argument on each side "
                        "with example opinions, notable misinformation, regional differences, authority response, and 2-3 "
                        "recommendations. Under 450 words.")
            go = True
    if go and question.strip():
        with st.status("Agent working…", expanded=True) as status:
            try:
                ctx = resources.analyst_context()
                run = run_agent(resources.llm(), ctx, question.strip(), on_step=lambda s: _render_step(s, st))
            except LLMError as exc:
                status.update(label="Agent failed", state="error")
                st.error(f"The model API refused the request (free-tier limits?): {exc}")
                return
            status.update(label=f"Done: {len(run.steps)} tool calls · {run.llm_calls} LLM calls · "
                                f"{run.tokens:,} tokens · {run.seconds}s", state="complete", expanded=False)
        answer, sources = link_citations(run.answer, ctx)
        st.session_state["agent_run"] = (answer, sources, run)
    if st.session_state.get("agent_run"):
        answer, sources, run = st.session_state["agent_run"]
        st.markdown(answer)
        if sources:
            st.markdown("**Sources**\n" + "\n".join(f"{n}. [{title}]({url})" for n, title, url in sources))
        with st.expander(f"Reasoning trace ({len(run.steps)} tool calls)"):
            for i, step in enumerate(run.steps, 1):
                st.markdown(f"**Step {i}: `{step.tool}`** · {step.latency_ms:.0f} ms")
                st.json({"arguments": step.arguments, "result": step.result if not step.error else {"error": step.error}},
                        expanded=False)
