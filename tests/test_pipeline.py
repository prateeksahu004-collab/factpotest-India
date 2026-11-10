import json
import re

import numpy as np

from factprotest.agent import AnalystContext, OpinionIndex, link_citations, run_agent
from factprotest.factcheck import OpinionInput, assess
from factprotest.llm import LLMClient
from tests.fakes import FakeGroq, HashEncoder, completion, tool_call
from tests.test_rag import _retriever


def factcheck_responder(kwargs):
    system, user = kwargs["messages"][0]["content"], kwargs["messages"][1]["content"]
    if "fact-checker" in system:
        verdicts = [{"claim_id": cid, "verdict": "REFUTED" if "2019" in text else "SUPPORTED", "confidence": 0.9,
                     "evidence_ids": [f"{cid}.E1"], "explanation": "test"}
                    for cid, text in re.findall(r"### Claim (C\d+): (.+)", user)]
        return completion(json.dumps({"verdicts": verdicts}))
    results = [{"opinion_id": oid, "claims": [body.strip().split(". ")[0]], "reasoning_quality": 4, "specificity": 4,
                "civility": 5, "emotional_intensity": 2, "predicted_stance": "Oppose", "rationale": "test"}
               for oid, body in re.findall(r'<opinion id="([^"]+)"[^>]*>\n(.+?)\n</opinion>', user, re.S)]
    return completion(json.dumps({"results": results}))


def test_assess_end_to_end_with_fake_models():
    llm = LLMClient(client=FakeGroq(factcheck_responder))
    items = [OpinionInput("a", 1, "Farm laws", "Support", "The farm laws were repealed in November 2021. Good."),
             OpinionInput("b", 1, "Farm laws", "Oppose", "The farm laws were repealed in 2019. Bad.")]
    out = assess(llm, _retriever(), items)
    assert out["a"].claims[0].verdict == "SUPPORTED" and out["b"].claims[0].verdict == "REFUTED"
    assert out["a"].claims[0].evidence[0].chunk_id.startswith("P1-")
    assert out["b"].score.final < out["a"].score.final
    assert "stance_mismatch" in out["a"].score.flags and out["a"].stance_consistent is False
    assert len(llm.records) == 2


def _context() -> AnalystContext:
    opinions = [{"id": f"op{i}", "issue_id": "1", "stance": s, "state": "Punjab", "opinion_text": t,
                 "credibility_score": 60, "ai_analysis": None, "created_at": "2026-11-01T00:00:00"}
                for i, (s, t) in enumerate([("Support", "MSP must be legal"), ("Oppose", "reforms were needed")] * 3)]
    encoder = HashEncoder()
    return AnalystContext(
        protests=[{"id": 1, "title": "Farm laws", "status": "Resolved"}], opinions=opinions, replies=[],
        retriever=_retriever(),
        opinion_index=OpinionIndex([o["id"] for o in opinions], encoder.documents([o["opinion_text"] for o in opinions]),
                                   lambda q: encoder.queries([q])[0]))


def test_agent_calls_tools_then_answers():
    replies = iter([
        completion(tool_calls=[tool_call("t1", "get_protest_stats", '{"protest_id": 1}'),
                               tool_call("t2", "search_wikipedia", '{"query": "repealed", "protest_id": 1}')]),
        completion(tool_calls=[tool_call("t3", "not_a_tool", "{}")]),
        completion("Support is 50% [W:P1-x-000]."),
    ])
    fake = FakeGroq(lambda kwargs: next(replies))
    ctx = _context()
    run = run_agent(LLMClient(client=fake), ctx, "Summarise protest 1")
    assert [s.tool for s in run.steps] == ["get_protest_stats", "search_wikipedia", "not_a_tool"]
    assert run.steps[0].result["stance_counts"] == {"Oppose": 3, "Neutral": 0, "Support": 3}
    assert run.steps[2].error == "unknown tool not_a_tool"
    tool_messages = [m for m in fake.calls[-1]["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 3
    linked, sources = link_citations(run.answer, ctx)
    assert "[[1]](https://example.org)" in linked and sources[0][0] == 1


def test_agent_bad_arguments_return_error_not_crash():
    replies = iter([completion(tool_calls=[tool_call("t1", "get_protest_stats", '{"protest_id": 42}')]),
                    completion("No such protest.")])
    run = run_agent(LLMClient(client=FakeGroq(lambda kwargs: next(replies))), _context(), "q")
    assert "unknown protest_id" in run.steps[0].error
    assert np.isclose(len(run.steps), 1)
