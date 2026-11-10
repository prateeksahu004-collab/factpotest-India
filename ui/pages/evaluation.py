from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import resources
from ui.style import kpis, section

ARCHITECTURE = """
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#94a3b8", fontname="Helvetica", fontsize=11];
  edge [color="#64748b", fontname="Helvetica", fontsize=9];
  user [label="Citizen opinion", fillcolor="#e0f2fe"];
  guard [label="Safety\\nPrompt Guard 2 (injection)\\ngpt-oss-safeguard (policy)"];
  analyze [label="Claim extraction + rubric\\n+ stance (gpt-oss-20b,\\nJSON schema)"];
  kb [label="Wikipedia KB\\n17 articles → 760 passages\\nbge-small embeddings", fillcolor="#fef3c7"];
  retr [label="Hybrid retrieval\\nBM25 + dense → RRF"];
  verify [label="Evidence-grounded verifier\\n(gpt-oss-120b)\\nSUPPORTED / REFUTED / NEI"];
  score [label="Explainable score\\nevidence 40% · reasoning 40%\\n· civility 20%", fillcolor="#dcfce7"];
  db [label="Supabase\\n(Postgres)", fillcolor="#ede9fe"];
  ml [label="Offline ML\\nstance classifier · topics\\n· coordination detector"];
  agent [label="Civic Analyst agent\\n(tool calling, 7 tools)", fillcolor="#fee2e2"];
  user -> guard -> analyze -> retr -> verify -> score -> db;
  kb -> retr; db -> ml; db -> agent; kb -> agent; ml -> agent;
}
"""


def _fmt(value: float | None, spec: str) -> str:
    return "–" if value is None or value != value else format(value, spec)


def _table(rows: dict[str, dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{"config": k, **{c: v.get(c) for c in columns}} for k, v in rows.items()]).set_index("config")


def page_evaluation() -> None:
    st.title("🧪 Model Card & Evaluation")
    st.caption("Every number on this page is produced by `scripts/run_evals.py` and `scripts/train_models.py` "
               "and committed to the repo under `evals/results/` and `artifacts/metrics/`.")
    st.graphviz_chart(ARCHITECTURE, use_container_width=True)
    summary = resources.eval_summary()
    metrics = resources.ml_metrics()
    if not summary:
        st.warning("No evaluation results found. Run `python scripts/run_evals.py <suite>` first.")
        return
    verification = summary.get("verification", {}).get("configs", {})
    prod_name = next((n for n in ("hybrid_k3__gpt-oss-120b", "hybrid_k3__gpt-oss-20b") if n in verification), None)
    prod = verification.get(prod_name, {}) if prod_name else {}
    e2e = summary.get("endtoend", {})
    retrieval = summary.get("retrieval", {}).get("results", {})
    stance = metrics.get("stance", {}).get("models", {})
    stance_llm = summary.get("stance_llm", {})
    label_audit = summary.get("stance_label_audit", {})
    integ = metrics.get("integrity", {})
    paraphrased_hybrid = summary.get("retrieval_paraphrase", {}).get("results", {}).get("hybrid", {})
    kpis([
        ("Claim verification macro-F1", _fmt(prod.get("macro_f1"), ".2f"),
         f"{prod.get('model', '').split('/')[-1]} · n={prod.get('n', '–')} gold claims"),
        ("Verification accuracy", _fmt(prod.get("accuracy"), ".1%"), "SUPPORTED / REFUTED / NEI"),
        ("Retrieval MRR", _fmt(paraphrased_hybrid.get("mrr@10"), ".2f"), "hybrid, paraphrased claims"),
        ("Stance macro-F1 (LLM)", _fmt(stance_llm.get("vs_audited_labels", {}).get("macro_f1"), ".2f"),
         f"target-aware prompt · n={stance_llm.get('n', '–')}"),
        ("Campaign detection F1", _fmt(integ.get("threshold_sweep", {}).get("0.88", {}).get("f1"), ".2f"),
         "post-level, cosine ≥ 0.88"),
        ("Scorer stance bias", _fmt(e2e.get("score_bias_audit_by_declared_stance", {}).get("rank_biserial"), ".2f"),
         "rank-biserial r (≈0 = unbiased)"),
    ])

    section("1 · Retrieval: does the right evidence come back?")
    if retrieval:
        paraphrased = summary.get("retrieval_paraphrase", {}).get("results", {})
        cols = ["recall@1", "recall@3", "relaxed_recall@1", "mrr@10"]
        table = pd.DataFrame({f"{mode} · {name}": {c: res[mode][c] for c in cols}
                              for name, res in (("claims as written", retrieval), ("paraphrased claims", paraphrased)) if res
                              for mode in ("bm25", "dense", "hybrid")}).T
        st.dataframe(table, use_container_width=True)
        st.caption(f"{summary['retrieval'].get('n_queries')} gold claims with a known source passage, searched within the "
                   f"protest's slice of the {summary['retrieval'].get('kb_chunks')}-passage KB. When claims reuse the source "
                   "wording, BM25 alone is as good as hybrid; when they are paraphrased the way citizens actually write, "
                   "hybrid fusion is clearly best, which is why production uses it.")

    section("2 · Claim verification: accuracy by configuration")
    if verification:
        st.dataframe(_table(verification, ["model", "accuracy", "macro_f1", "tokens_per_claim", "missing"]),
                     use_container_width=True)
        oracle_note = ("Oracle = the gold passage is given directly, which isolates reasoning errors from retrieval errors. "
                       if any(name.startswith("oracle") for name in verification) else "")
        st.caption(oracle_note + "Gold claims were drafted by a different model family (Qwen) from source passages "
                   "and then reviewed.")
        cm = prod.get("confusion_matrix")
        if cm:
            fig = go.Figure(go.Heatmap(z=cm["matrix"], x=[f"pred {x}" for x in cm["labels"]], y=[f"true {y}" for y in cm["labels"]],
                                       colorscale="Blues", text=cm["matrix"], texttemplate="%{text}", showscale=False))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title=f"Confusion matrix · {prod_name}")
            st.plotly_chart(fig, use_container_width=True)
            per_class = pd.DataFrame(prod["per_class"]).T[["precision", "recall", "f1-score", "support"]]
            st.dataframe(per_class, use_container_width=True)

    section("3 · End-to-end on the synthetic corpus with known ground truth")
    if e2e:
        mis = e2e["misinformation"]
        rlo, rhi = mis.get("recall_wilson_95ci", [None, None])
        flo, fhi = mis.get("false_alarm_wilson_95ci", [None, None])
        kpis([("Opinions scored", str(e2e["n_scored"]), f"protests {', '.join(map(str, e2e.get('protests_covered', [])))}"),
              ("Misinformation recall", f"{mis['recall_refuted_flag']:.0%}", f"95% CI {rlo:.0%}–{rhi:.0%} · n={mis['opinions_with_injected_false_claim']}"),
              ("False-alarm rate", f"{mis['false_alarm_rate_accurate_fact']:.0%}", f"95% CI {flo:.0%}–{fhi:.0%} · n={mis.get('opinions_accurate_fact')}"),
              ("False alarms on fact-free text", f"{mis['false_alarm_rate_no_fact']:.0%}", "pure opinions"),
              ("Mean score", f"{mis['mean_score']['accurate_fact']} vs {mis['mean_score']['false_fact']}", "accurate vs false facts")])
        st.markdown(
            "**Error analysis.** 7 of the 8 'false alarms' are one recurring claim: *the farm laws were repealed on "
            "19 November 2021*. Wikipedia records that date as the **announcement**; Parliament passed the repeal on 29 November "
            "and it took effect on 1 December. The verifier is technically right, so the gap is a policy question (should "
            "announced-vs-enacted count as false?), which motivates a future PARTIALLY_ACCURATE verdict. The overall score "
            "separates accurate from false opinions only weakly (ROC-AUC "
            f"{mis['auc_score_accurate_vs_false']:.2f}) because 60% of it rates reasoning and civility, by design.")
        audit = e2e.get("score_bias_audit_by_declared_stance", {})
        if audit:
            st.caption(f"Stance-bias audit: mean score by declared stance {audit.get('means')}; Kruskal–Wallis "
                       f"p = {audit.get('kruskal_p', float('nan')):.3g}; Support-vs-Oppose rank-biserial r = {audit.get('rank_biserial')} "
                       "(negligible): the score does not favour either side.")

    section("4 · Stance classification (Support / Oppose / Neutral)")
    if label_audit:
        st.markdown(
            f"**Label audit first.** A manual audit found the corpus generator had labelled stance toward the *policy* rather "
            f"than the *protest* (its labels matched the audit {label_audit['generator_vs_manual_accuracy']:.0%} of the time). "
            f"Labels were re-annotated by a second model family given each protest's explicit demand and validated against "
            f"the audit: {label_audit['silver_vs_manual_accuracy']:.0%} agreement, Cohen's κ = {label_audit['silver_vs_manual_kappa']:.2f} "
            f"(n = {label_audit['manual_audit_n']}).")
    rows = {k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"], "protocol": "leave-one-protest-out"} for k, v in stance.items()}
    if stance_llm:
        rows[f"LLM zero-shot, target-aware ({stance_llm['model'].split('/')[-1]})"] = {
            "accuracy": stance_llm["vs_audited_labels"]["accuracy"], "macro_f1": stance_llm["vs_audited_labels"]["macro_f1"],
            "protocol": "zero-shot"}
    v2 = e2e.get("llm_stance_from_analyze_v2_prompt")
    if v2:
        rows["LLM, prompt without the demand (analyze/v2)"] = {"accuracy": v2["accuracy"], "macro_f1": v2["macro_f1"],
                                                              "protocol": f"zero-shot, n={v2['n']}"}
    if rows:
        st.dataframe(pd.DataFrame(rows).T, use_container_width=True)
        st.caption("Classical models trained on 400 examples generalise poorly to unseen protests (in-domain random-split "
                   "macro-F1 is ~0.54), while telling the LLM exactly what the protesters demand lifts its macro-F1 from 0.43 "
                   "to 0.75. Production therefore uses the target-aware LLM prompt (analyze/v3); the classifier is a free, "
                   "instant second opinion in the UI.")

    section("5 · Coordinated-campaign detection")
    if integ:
        sweep = integ.get("threshold_sweep", {})
        fig = go.Figure()
        for metric, color in (("precision", "#2563eb"), ("recall", "#16a34a"), ("f1", "#dc2626")):
            fig.add_scatter(x=[float(t) for t in sweep], y=[v[metric] for v in sweep.values()], name=metric,
                            mode="lines+markers", line=dict(color=color))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="cosine threshold",
                          yaxis=dict(range=[0, 1.05]), title="Threshold sweep (≥3 posts, same protest, within 3 h)")
        st.plotly_chart(fig, use_container_width=True)
        if integ.get("ablation_without_time_window"):
            st.caption(f"Ablation without the time window: {integ['ablation_without_time_window']}")
        if integ.get("legacy_v1_seed_audit"):
            st.caption(f"Audit of the v1 demo data: {integ['legacy_v1_seed_audit']}")

    section("6 · Stability")
    cons = summary.get("consistency")
    if cons:
        st.caption(f"Re-scoring {cons['opinions']} opinions {cons['runs_per_opinion']}×: mean score std "
                   f"{cons['mean_score_std']} points, median range {cons['median_score_range']}, "
                   f"{cons['share_within_10_points']:.0%} stay within 10 points; predicted stance identical across runs "
                   f"for {cons['stance_agreement_all_runs']:.0%}.")

    section("Data, limitations and responsible use")
    st.markdown(
        "- **Knowledge base**: 17 English Wikipedia articles (CC BY-SA 4.0), revision ids pinned in `data/kb/sources.json`. "
        "Wikipedia can lag or be contested; a claim it does not cover is NOT_ENOUGH_INFO, never REFUTED.\n"
        "- **Demo opinions are synthetic**: generated from persona specs by an LLM so that stance labels, injected false "
        "claims and coordinated campaigns have ground truth. Metrics on them measure the pipeline, not Indian public opinion.\n"
        "- **Credibility ≠ truth of a viewpoint**: the score rates evidence, reasoning and civility. Opinions without "
        "checkable facts are scored on reasoning and civility alone.\n"
        "- **Free-tier LLMs**: Groq free tier rate limits cap throughput; the pipeline batches calls and rate-limits per model.")
