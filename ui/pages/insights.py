from __future__ import annotations

from collections import Counter
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from factprotest import analytics
from factprotest.data import parse_analysis
from ui import resources
from ui.style import STANCE_COLOR, VERDICT_COLOR, esc, kpis, section

LAYOUT = dict(plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff", font=dict(family="Inter, Arial", size=12),
              margin=dict(l=10, r=10, t=40, b=10))


@st.cache_data(show_spinner="Bootstrapping confidence intervals…")
def _summaries(rows_key: str, rows: list[dict[str, Any]], protest_ids: tuple[str, ...]) -> dict[str, Any]:
    return {pid: analytics.protest_summary([o for o in rows if str(o["issue_id"]) == pid]) for pid in protest_ids}


def _fmt_ci(triple: tuple[float, float, float], pct: bool = False) -> tuple[str, str]:
    point, lo, hi = triple
    if pct:
        return f"{point:.0%}", f"95% CI {lo:.0%}–{hi:.0%}"
    return f"{point:.2f}" if point < 1.5 else f"{point:.0f}", f"95% CI {lo:.2f}–{hi:.2f}" if hi < 1.5 else f"95% CI {lo:.0f}–{hi:.0f}"


def page_insights() -> None:
    st.title("📊 Data Insights")
    st.caption("Statistics with uncertainty, not just counts: bootstrap confidence intervals, Leik's ordinal polarization "
               "index, chi-square tests of regional differences, an audit of the credibility scorer for stance bias, "
               "ML-discovered themes and coordinated-campaign detection.")
    protests = resources.protests()
    rows = resources.opinions()
    if not rows:
        st.info("No opinions yet.")
        return
    titles = {str(p["id"]): p["title"] for p in protests}
    key = f"{len(rows)}-{rows[0]['id']}"
    summaries = _summaries(key, rows, tuple(titles))

    section("Consensus map: where does India agree, and where is it split?")
    pts = [(pid, s) for pid, s in summaries.items() if s["n"]]
    fig = go.Figure(go.Scatter(
        x=[s["support_share"][0] for _, s in pts], y=[s["polarization"][0] for _, s in pts],
        error_y=dict(type="data", symmetric=False, array=[s["polarization"][2] - s["polarization"][0] for _, s in pts],
                     arrayminus=[s["polarization"][0] - s["polarization"][1] for _, s in pts], color="#94a3b8"),
        mode="markers+text", text=[titles[pid].split("—")[0].strip()[:28] for pid, _ in pts], textposition="top center",
        marker=dict(size=[8 + s["n"] / 3 for _, s in pts], color=[s["mean_credibility"][0] for _, s in pts],
                    colorscale="RdYlGn", cmin=30, cmax=80, showscale=True, colorbar=dict(title="Mean<br>credibility")),
        hovertemplate="%{text}<br>support %{x:.0%}<br>polarization %{y:.2f}<extra></extra>"))
    fig.update_layout(**LAYOUT, height=430, xaxis=dict(title="Support share", tickformat=".0%", range=[0, 1.05]),
                      yaxis=dict(title="Polarization (Leik's D: 0 = consensus, 1 = split between extremes)", range=[0, 1.05]))
    st.plotly_chart(fig, use_container_width=True)

    choice = st.selectbox("Drill into", ["All protests"] + [titles[pid] for pid in titles])
    pid = next((k for k, v in titles.items() if v == choice), None)
    subset = rows if pid is None else [o for o in rows if str(o["issue_id"]) == pid]
    summary = analytics.protest_summary(subset) if pid is None else summaries[pid]
    analyses = [parse_analysis(o) for o in subset]
    checked = [a for a in analyses if a.get("schema_version") == 2]
    refuted = sum(any(c["verdict"] == "REFUTED" for c in a["claims"]) for a in checked)
    kpis([("Opinions", str(summary["n"]), f"{len(checked)} fact-checked"),
          ("Support share", *_fmt_ci(summary["support_share"], pct=True)),
          ("Polarization (Leik D)", *_fmt_ci(summary["polarization"])),
          ("Mean credibility", *_fmt_ci(summary["mean_credibility"])),
          ("Refuted-claim rate", f"{refuted / max(len(checked), 1):.0%}", "opinions with ≥1 refuted claim")])

    left, right = st.columns(2)
    with left:
        section("Does stance depend on the state?")
        states = [o.get("state") or "Not Specified" for o in subset]
        stances = [o["stance"] for o in subset]
        test = analytics.stance_by_state_test(states, stances, min_count=5 if pid else 10)
        top_states = [s for s, n in Counter(states).most_common(10) if s != "Not Specified" and n >= (5 if pid else 10)]
        fig = go.Figure()
        for stance in ("Support", "Neutral", "Oppose"):
            fig.add_bar(y=top_states, x=[sum(1 for st_, sn in zip(states, stances) if st_ == s and sn == stance) for s in top_states],
                        name=stance, orientation="h", marker_color=STANCE_COLOR[stance])
        fig.update_layout(**LAYOUT, barmode="stack", barnorm="percent", height=320, legend=dict(orientation="h", y=1.12),
                          xaxis=dict(title="% of opinions"))
        st.plotly_chart(fig, use_container_width=True)
        if test.get("testable"):
            verdict = "significant" if test["p_value"] < 0.05 else "not significant"
            st.caption(f"χ²({test['dof']}) = {test['chi2']}, p = {test['p_value']:.3g} ({verdict} at α=0.05), "
                       f"Cramér's V = {test['cramers_v']} on n = {test['n']}. "
                       + (f"{test['low_expected_cells']} cells have expected count < 5, so treat this as indicative."
                          if test["low_expected_cells"] else ""))
    with right:
        section("Is the credibility scorer biased toward one side?")
        audit = analytics.score_bias_audit([o.get("credibility_score") for o in subset], [o["stance"] for o in subset])
        fig = go.Figure()
        for stance in ("Support", "Neutral", "Oppose"):
            fig.add_box(y=[o["credibility_score"] for o in subset if o["stance"] == stance and o.get("credibility_score") is not None],
                        name=stance, marker_color=STANCE_COLOR[stance], boxmean=True)
        fig.update_layout(**LAYOUT, height=320, showlegend=False, yaxis=dict(title="Credibility score"))
        st.plotly_chart(fig, use_container_width=True)
        if "kruskal_p" in audit:
            effect = audit.get("rank_biserial")
            size = "negligible" if effect is not None and abs(effect) < 0.1 else "small" if effect is not None and abs(effect) < 0.3 else "notable"
            st.caption(f"Kruskal–Wallis H = {audit['kruskal_h']}, p = {audit['kruskal_p']:.3g}. Support vs Oppose: "
                       f"Mann–Whitney p = {audit.get('mannwhitney_p', float('nan')):.3g}, "
                       f"rank-biserial r = {effect} ({size} effect). A fair scorer should judge reasoning and evidence, "
                       "not which side someone is on.")

    section("Fact-check outcomes")
    verdicts = Counter(c["verdict"] for a in checked for c in a["claims"])
    c1, c2 = st.columns(2)
    fig = go.Figure(go.Bar(x=list(verdicts), y=list(verdicts.values()),
                           marker_color=[VERDICT_COLOR[v] for v in verdicts]))
    fig.update_layout(**LAYOUT, height=280, title="Verdicts on extracted claims")
    c1.plotly_chart(fig, use_container_width=True)
    flags = Counter(f for a in checked for f in a["score"]["flags"])
    fig = go.Figure(go.Bar(y=list(flags), x=list(flags.values()), orientation="h", marker_color="#475569"))
    fig.update_layout(**LAYOUT, height=280, title="Pipeline flags")
    c2.plotly_chart(fig, use_container_width=True)

    if pid and resources.topics().get(pid):
        section("Themes (KMeans on bge embeddings, k by silhouette, keywords by c-TF-IDF, labels by LLM)")
        for t in resources.topics()[pid]:
            mix = ", ".join(f"{k}: {v}" for k, v in t["stance_mix"].items())
            st.markdown(f"**{t['label']}** · {t['size']} opinions · {mix}  \n_{t['summary']}_  \n"
                        f"Keywords: `{'`, `'.join(t['keywords'][:6])}`")

    section("Coordinated-campaign alerts")
    index = resources.opinion_index(rows)
    clusters = [c for c in resources.coordination(rows, index) if pid is None or str(c["protest_id"]) == pid]
    if not clusters:
        st.success("No coordinated clusters detected (≥3 near-identical posts within 3 hours).")
    by_id = {str(o["id"]): o for o in rows}
    for c in clusters:
        with st.expander(f"⚠️ {len(c['member_ids'])} near-identical posts on "
                         f"{titles.get(str(c['protest_id']), c['protest_id'])[:50]} · mean cosine {c['mean_similarity']} · "
                         f"within {c['span_minutes']:.0f} min · {c['distinct_authors']} accounts"):
            for mid in c["member_ids"]:
                o = by_id.get(mid)
                if o:
                    st.html(f'<div class="muted">{esc(o.get("user_name"))} · {esc(str(o["created_at"])[:16])}</div>'
                            f'<div style="font-size:0.85rem;margin-bottom:6px">{esc(o["opinion_text"])}</div>')
