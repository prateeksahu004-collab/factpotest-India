"""FactProtest India: Streamlit entry point. All logic lives in `factprotest/` (core) and `ui/` (pages)."""
from __future__ import annotations

import streamlit as st

from ui import resources, style
from ui.pages.analyst import page_analyst
from ui.pages.authority import page_authority, page_replies
from ui.pages.evaluation import page_evaluation
from ui.pages.home import page_detail, page_home
from ui.pages.insights import page_insights
from ui.pages.lab import page_lab

PAGES = {
    "🏠 Protests": "home",
    "🔬 Fact-Check Lab": "lab",
    "🤖 AI Analyst": "analyst",
    "📊 Data Insights": "insights",
    "🧪 Model Card & Evals": "evaluation",
    "🏛 Authority Replies": "replies",
    "🔐 Authority Portal": "authority",
}
RENDER = {"home": page_home, "detail": page_detail, "lab": page_lab, "analyst": page_analyst,
          "insights": page_insights, "evaluation": page_evaluation, "replies": page_replies, "authority": page_authority}


def main() -> None:
    st.set_page_config(page_title="FactProtest India", page_icon="🇮🇳", layout="wide", initial_sidebar_state="expanded")
    style.inject()
    st.session_state.setdefault("page", "home")
    with st.sidebar:
        st.html('<div style="padding:16px 8px 12px;border-bottom:1px solid #1e293b;margin-bottom:8px">'
                '<div style="font-size:1.12rem;font-weight:800">🇮🇳 FactProtest India</div>'
                '<div style="font-size:0.72rem;color:#94a3b8">Claim-level fact-checking · AI agent · civic analytics</div></div>')
        choice = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
        mode = "local demo data" if resources.data_mode() == "local" else "Supabase"
        st.html(f'<div style="margin-top:24px;font-size:0.7rem;color:#64748b">Groq · gpt-oss-20b/120b · Llama Prompt Guard 2<br>'
                f'bge-small embeddings · Wikipedia KB<br>Data: {mode}</div>')
    target = PAGES[choice]
    if target != "home" or st.session_state["page"] not in ("home", "detail"):
        st.session_state["page"] = target
    RENDER[st.session_state["page"]]()


if __name__ == "__main__":
    main()
