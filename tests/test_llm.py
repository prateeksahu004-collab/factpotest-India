import json

import pytest
from pydantic import BaseModel

from factprotest.llm import LLMClient, LLMError, parse_json_object
from tests.fakes import FakeGroq, completion


class Answer(BaseModel):
    value: int


def test_parse_json_object_handles_fences_and_prose():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Here you go: {"a": {"b": 2}} done') == {"a": {"b": 2}}
    with pytest.raises(json.JSONDecodeError):
        parse_json_object("no json here")


def test_json_repairs_invalid_output_once():
    replies = iter(['{"value": "not a number"}', '{"value": 7}'])
    fake = FakeGroq(lambda kwargs: completion(next(replies)))
    client = LLMClient(client=fake)
    assert client.json(system="s", user="u", schema=Answer, model="openai/gpt-oss-20b", purpose="t").value == 7
    assert len(fake.calls) == 2
    assert "failed validation" in fake.calls[1]["messages"][-1]["content"]
    assert fake.calls[0]["reasoning_effort"] == "low"


class RateLimited(Exception):
    status_code = 429


def test_retries_rate_limits_then_succeeds(monkeypatch):
    monkeypatch.setattr("factprotest.llm.time.sleep", lambda s: None)
    attempts = {"n": 0}

    def responder(kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RateLimited("slow down")
        return completion('{"value": 1}')

    client = LLMClient(client=FakeGroq(responder))
    assert client.json(system="s", user="u", schema=Answer, model="qwen/x", purpose="t").value == 1
    assert client.retries == 2


def test_non_retryable_errors_raise():
    class BadRequest(Exception):
        status_code = 400

    client = LLMClient(client=FakeGroq(lambda kwargs: (_ for _ in ()).throw(BadRequest("nope"))))
    with pytest.raises(LLMError):
        client.text(system="s", user="u", model="m", purpose="t")
    assert client.usage_summary()["failed"] == 1


def test_huge_retry_after_is_capped(monkeypatch):
    from types import SimpleNamespace

    from factprotest.llm import MAX_RETRY_WAIT

    sleeps: list[float] = []
    monkeypatch.setattr("factprotest.llm.time.sleep", sleeps.append)

    class DailyQuota(Exception):
        status_code = 429
        response = SimpleNamespace(headers={"retry-after": "4000"})

    replies = iter([DailyQuota("quota"), completion('{"value": 3}')])

    def responder(kwargs):
        item = next(replies)
        if isinstance(item, Exception):
            raise item
        return item

    client = LLMClient(client=FakeGroq(responder))
    assert client.json(system="s", user="u", schema=Answer, model="m", purpose="t").value == 3
    assert sleeps == [MAX_RETRY_WAIT]


def test_rate_limiter_blocks_until_window_frees(monkeypatch):
    from factprotest.llm import TokenRateLimiter

    clock = {"t": 1000.0}
    monkeypatch.setattr("factprotest.llm.time.monotonic", lambda: clock["t"])
    monkeypatch.setattr("factprotest.llm.time.sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    limiter = TokenRateLimiter(tokens_per_minute=1000)
    limiter.record("m", 900)
    limiter.wait("m", 50)
    assert clock["t"] == 1000.0
    limiter.wait("m", 500)
    assert clock["t"] >= 1060.0
