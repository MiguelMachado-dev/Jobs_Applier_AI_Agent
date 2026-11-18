"""The model's verdict: the request's shape, the one question, the score from its certainty, one posting to a request, the client's retries, the failed verdicts and the ones not judged. No network: a scripted endpoint stands in."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI

from aihawk.config import DEFAULT_BASE_URL, DEFAULT_DECISION_MODEL, load_config
from aihawk.database import Job
from aihawk.scoring import CONCURRENT, DESCRIPTION_CHARS, NOT_JUDGED, fits, question, score

KEY = "sk-test-key"
JOB = Job("g:a:1", "Acme", "Python Engineer", "Rome", "u", "a", "Build Python services.")
ASKED = question("Senior Python work.", "Rome, Italy")


def client_with(handler: Any) -> AsyncOpenAI:
    """The client the session makes, with the endpoint behind it: the same retries and timeout as llm.make_client."""
    return AsyncOpenAI(base_url=DEFAULT_BASE_URL, api_key=KEY, max_retries=5, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def completion(content: Any) -> httpx.Response:
    """A chat completion whose message says `content`, as JSON."""
    message = {"role": "assistant", "content": json.dumps(content)}
    return httpx.Response(200, json={"id": "c", "object": "chat.completion", "created": 0, "model": "m", "choices": [{"index": 0, "finish_reason": "stop", "message": message}]})


def sent(request: httpx.Request) -> dict[str, Any]:
    """What the request's user message holds: the question, the CV and the postings."""
    return json.loads(json.loads(request.content)["messages"][1]["content"])


def answering(certainty: float, seen: list[httpx.Request]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return completion({"fit_probability": certainty})

    return handler


def run(client: AsyncOpenAI, jobs: list[Job], cv: str = "cv", why: Callable[[str], None] = lambda reason: None) -> list[float | None]:
    return asyncio.run(fits(client, DEFAULT_DECISION_MODEL, cv, jobs, ASKED, why))


def postings(count: int, description: str = "") -> list[Job]:
    """`count` postings, the n-th titled Engineer n."""
    return [Job(f"j{number:03d}", "Firm", f"Engineer {number}", "Rome", "u", "a", description) for number in range(count)]


def test_the_request_goes_to_chat_completions_with_the_key_the_question_and_a_strict_schema() -> None:
    seen: list[httpx.Request] = []
    assert run(client_with(answering(0.8, seen)), [JOB]) == [0.8]
    (request,) = seen
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer " + KEY
    body = json.loads(request.content)
    assert body["model"] == DEFAULT_DECISION_MODEL and body["reasoning_effort"] == "medium"
    schema = body["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"] == {"type": "object", "additionalProperties": False, "required": ["fit_probability"], "properties": {"fit_probability": {"type": "number"}}}
    assert sent(request) == {"question": ASKED, "candidate_cv": "cv", "job_posting": "Python Engineer at Acme. Location: Rome\n\nBuild Python services."}
    assert "Senior Python work." in ASKED and "Rome, Italy" in ASKED


def test_the_score_is_how_sure_the_yes_is_from_1_to_5() -> None:
    assert [score(value) for value in (0.0, 0.5, 0.85, 1.0)] == [1.0, 3.0, 4.4, 5.0]
    assert score(-0.2) == 1.0 and score(1.3) == 5.0


def test_a_retryable_failure_is_retried_by_the_client_and_then_answers() -> None:
    seen: list[httpx.Request] = []
    answer = answering(0.9, seen)

    def flaky(request: httpx.Request) -> httpx.Response:
        if not seen:
            seen.append(request)
            # A 503 that says to try again at once: the client retries it, as it does for every call the session makes.
            return httpx.Response(503, headers={"retry-after-ms": "1"}, json={"error": "busy"})
        return answer(request)

    assert run(client_with(flaky), [JOB]) == [0.9]
    assert len(seen) == 2


def test_a_failed_request_fails_its_posting_alone_and_the_person_is_told_why() -> None:
    seen: list[httpx.Request] = []
    said: list[str] = []

    def refusing(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(400, json={"error": {"message": "the request is wrong", "code": "invalid_request_error"}})

    assert run(client_with(refusing), postings(1), why=said.append) == [None]
    assert len(seen) == 1
    assert len(said) == 1 and said[0] == "The AI service did not accept the request."


@pytest.mark.parametrize("answer", [{"unexpected": 1}, {"fit_probability": "yes"}, {"fit_probability": 1.5}])
def test_an_answer_of_another_shape_is_a_failed_verdict(answer: dict[str, Any]) -> None:
    assert run(client_with(lambda request: completion(answer)), [JOB]) == [None]


def test_a_reply_with_no_content_is_a_failed_verdict_and_says_why() -> None:
    said: list[str] = []
    message = {"role": "assistant", "content": None, "refusal": "no"}
    reply = {"id": "c", "object": "chat.completion", "created": 0, "model": "m", "choices": [{"index": 0, "finish_reason": "stop", "message": message}]}
    assert run(client_with(lambda request: httpx.Response(200, json=reply)), [JOB], why=said.append) == [None]
    assert len(said) == 1


def test_each_posting_is_one_request_and_the_verdicts_come_back_in_order() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        # Each posting's answer differs, so a value in the wrong place shows.
        return completion({"fit_probability": int(re.search(r"Engineer (\d+) at", sent(request)["job_posting"])[1]) / 1000})

    assert run(client_with(handler), postings(45)) == pytest.approx([number / 1000 for number in range(45)])
    assert len(seen) == 45


def test_one_failed_posting_leaves_the_others_their_verdicts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if b"Engineer 7 at" in request.content:
            return httpx.Response(400, json={"error": {"message": "the request is wrong", "code": "invalid_request_error"}})
        return completion({"fit_probability": 0.9})

    values = run(client_with(handler), postings(12))
    assert values[7] is None and values[:7] + values[8:] == [0.9] * 11


def test_a_description_is_cut_to_the_limit() -> None:
    seen: list[httpx.Request] = []
    assert run(client_with(answering(0.7, seen)), postings(1, "x" * (DESCRIPTION_CHARS + 500))) == [0.7]
    assert sent(seen[0])["job_posting"].count("x") == DESCRIPTION_CHARS


def test_no_more_requests_than_the_limit_run_at_once() -> None:
    now = most = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal now, most
        now += 1
        most = max(most, now)
        await asyncio.sleep(0.01)
        now -= 1
        return completion({"fit_probability": 0.5})

    assert run(client_with(handler), postings(CONCURRENT * 5)) == [0.5] * (CONCURRENT * 5)
    assert most == CONCURRENT


def test_the_decision_model_is_settable_in_the_settings(tmp_path: Path) -> None:
    assert load_config(tmp_path).decision_model == "gpt-6.2-sol"
    (tmp_path / "aihawk.yaml").write_text("decision_model: other/model\n", encoding="utf-8")
    assert load_config(tmp_path).decision_model == "other/model"


def client_answering(status: int, body: Any) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=DEFAULT_BASE_URL, api_key=KEY, max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(status, json=body))))


@pytest.mark.parametrize("status", [429, 500, 503, 408])
def test_a_service_that_does_not_answer_leaves_the_postings_not_judged_and_says_nothing_of_them(status: int) -> None:
    said: list[str] = []
    assert run(client_answering(status, {"error": "busy"}), postings(3), why=said.append) == [NOT_JUDGED] * 3
    assert said == []


def test_a_lost_connection_leaves_the_postings_not_judged() -> None:
    def lost(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    client = AsyncOpenAI(base_url=DEFAULT_BASE_URL, api_key=KEY, max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(lost)))
    assert run(client, [JOB]) == [NOT_JUDGED]


@pytest.mark.parametrize("status", [402, 403, 404])
def test_what_is_about_the_account_and_not_about_a_posting_is_raised_and_no_posting_is_set_aside(status: int) -> None:
    from openai import APIStatusError

    with pytest.raises(APIStatusError) as raised:
        run(client_answering(status, {"error": "no"}), postings(3))
    assert raised.value.status_code == status


def test_an_openai_account_out_of_credit_is_raised_and_not_asked_again_as_a_busy_service() -> None:
    from openai import RateLimitError

    body = {"error": {"message": "You exceeded your current quota, please check your plan and billing details.", "type": "insufficient_quota", "code": "insufficient_quota"}}
    with pytest.raises(RateLimitError):
        run(client_answering(429, body), postings(3))


def test_an_account_with_no_credit_is_told_in_plain_words_and_what_to_do() -> None:
    from openai import APIStatusError, RateLimitError

    from aihawk.llm import explain

    request = httpx.Request("POST", "https://x.example")
    paid = "Your OpenAI account has no credit left. Add some at platform.openai.com/settings/organization/billing, then press → to go on."
    assert explain(APIStatusError("Error code: 402", response=httpx.Response(402, request=request), body=None)) == paid
    quota = RateLimitError("Error code: 429", response=httpx.Response(429, request=request), body={"code": "credit_balance_exhausted", "message": "quota"})
    assert explain(quota) == paid


def test_a_refused_account_ends_the_round_at_once_and_the_requests_not_yet_made_are_not_made() -> None:
    from openai import RateLimitError

    seen: list[httpx.Request] = []
    body = {"error": {"message": "Your organization has no prepaid credits remaining.", "type": "insufficient_quota", "code": "credit_balance_exhausted"}}

    def broke(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(429, json=body)

    client = AsyncOpenAI(base_url=DEFAULT_BASE_URL, api_key=KEY, max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(broke)))
    with pytest.raises(RateLimitError):
        run(client, postings(CONCURRENT * 5))
    # The ones already under way when the first refusal came may finish; the rest of the round is never asked.
    assert len(seen) <= CONCURRENT


@pytest.mark.parametrize(
    ("code", "kind", "named"),
    [
        ("credit_balance_exhausted", "insufficient_quota", "no_credit"),
        ("insufficient_quota", "insufficient_quota", "no_credit"),
        ("", "insufficient_quota", "no_credit"),
        ("organization_spend_limit_exceeded", "insufficient_quota", "spend_limit"),
        ("project_spend_limit_exceeded", "insufficient_quota", "spend_limit"),
        ("organization_usage_limit_exceeded", "insufficient_quota", "budget"),
        ("slow_down", "rate_limit_error", ""),
        ("rate_limit_exceeded", "requests", ""),
    ],
)
def test_openais_429s_about_money_are_about_the_account_and_the_others_a_busy_service(code: str, kind: str, named: str) -> None:
    from openai import RateLimitError

    from aihawk.llm import account_problem, cause, no_credit, not_answering

    error = RateLimitError("Error code: 429", response=httpx.Response(429, request=httpx.Request("POST", "https://x.example")), body={"error": {"message": "m", "type": kind, "code": code}})
    assert cause(error) == named
    assert account_problem(error) is bool(named) and not_answering(error) is not bool(named)
    assert no_credit(error) is (named == "no_credit")
