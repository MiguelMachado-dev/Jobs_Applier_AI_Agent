"""The model's verdict: the request's shape, the one question, the score from its certainty, one posting to a request, the client's retries and the failed verdicts. No network: a scripted endpoint stands in."""

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
from aihawk.shortlist import CONCURRENT, DESCRIPTION_CHARS, fits, question, score

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
    return asyncio.run(fits(client, DEFAULT_DECISION_MODEL, cv, jobs, ASKED, lambda done: None, why))


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
    assert body["model"] == DEFAULT_DECISION_MODEL
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
    assert len(said) == 1 and said[0].startswith("OpenAI rejected the request.")


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
    assert load_config(tmp_path).decision_model == "gpt-6.1-sol"
    (tmp_path / "aihawk.yaml").write_text("decision_model: other/model\n", encoding="utf-8")
    assert load_config(tmp_path).decision_model == "other/model"
