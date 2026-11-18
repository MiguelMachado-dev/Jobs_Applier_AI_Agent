"""The one model gateway, OpenAI: its chat and Responses APIs, the cost of its calls and how a failed call is told."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from aihawk.config import DEFAULT_BASE_URL, KEY_ENV, KEY_URL, REASONING_EFFORT, Config, ConfigError

# OpenAI's /models gives no prices, so the default models' list prices per token: prompt, completion, cached prompt. From
# its pricing page on 2026-11-10.
PRICES = {
    "gpt-6.1-sol": (2.00e-6, 10.00e-6, 0.10e-6),
    "gpt-5.4-mini": (0.75e-6, 4.50e-6, 0.075e-6),
    "gpt-5.4-nano": (0.20e-6, 1.25e-6, 0.02e-6),
}

# The SDK's own default is ten minutes a try, and a model that hangs would hold a request that long.
TIMEOUT = 180.0


# A spend's total and tokens at one moment: see Spend.mark.
Mark = tuple[float, dict[str, tuple[int, int]]]


class Spend:
    """What the model calls cost, in dollars: each call's tokens at the gateway's own price for its model, added up. The
    tokens each model took in and gave out are counted too, every call, priced or not.

    The prices come from PRICES and from the gateway's /models list where it gives any, and give each call's cost the moment it
    ends. A gateway that keeps a record of what it charged is read by `settle`, which corrects the total. A model neither prices
    is counted as a call of unknown cost.
    """

    def __init__(self) -> None:
        self.prices: dict[str, tuple[float, float, float]] = dict(PRICES)  # per token: prompt, completion, cached prompt
        self.total = 0.0
        self.unpriced = 0
        self.loaded = False
        self.pending: list[tuple[str, float]] = []  # the gateway's id of each call not yet settled, and its estimate
        # The tokens each model took in and gave out.
        self.tokens: dict[str, tuple[int, int]] = {}

    async def load_prices(self, client: AsyncOpenAI) -> None:
        """The price list, once. A gateway that does not give one leaves every call unpriced; nothing else stops."""
        if self.loaded:
            return
        self.loaded = True
        try:
            listed = await client.get("/models", cast_to=dict)
        except Exception:  # noqa: BLE001  The list is a convenience: a search or an application must not wait on it.
            return
        for model in (listed.get("data") or []) if isinstance(listed, dict) else []:
            pricing = model.get("pricing") or {}
            try:
                self.prices[model["id"]] = (float(pricing["prompt"]), float(pricing["completion"]), float(pricing.get("input_cache_read") or pricing["prompt"]))
            except (KeyError, TypeError, ValueError):
                continue

    def note(self, model: str, usage: Any, request_id: str = "") -> float:
        """One call's cost, estimated from its usage and added to the total; its gateway id kept for settling. Zero, and one more
        unpriced call, when the price is unknown."""
        if usage is None:
            # A client that says nothing about tokens: nothing to count or estimate, and nothing to call unpriced.
            if request_id:
                self.pending.append((request_id, 0.0))
            return 0.0
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        taken, given = self.tokens.get(model, (0, 0))
        self.tokens[model] = (taken + prompt, given + completion)
        estimate = 0.0
        if (price := self.prices.get(model)) is not None:
            details = getattr(usage, "prompt_tokens_details", None)
            cached = int(getattr(details, "cached_tokens", 0) or 0)
            estimate = (prompt - cached) * price[0] + cached * price[2] + completion * price[1]
            self.total += estimate
        else:
            self.unpriced += 1
        if request_id:
            self.pending.append((request_id, estimate))
        return estimate

    def mark(self) -> Mark:
        """Where the count stands now, to ask later what came after it."""
        return self.total, dict(self.tokens)

    def since(self, mark: Mark) -> tuple[float, dict[str, tuple[int, int]]]:
        """What the calls after `mark` cost, and the tokens each model took in and gave out in them; a model that took none is left out."""
        total, before = mark
        used = {}
        for model, (taken, given) in self.tokens.items():
            taken_before, given_before = before.get(model, (0, 0))
            if (taken, given) != (taken_before, given_before):
                used[model] = (taken - taken_before, given - given_before)
        return self.total - total, used

    async def settle(self, client: AsyncOpenAI) -> None:
        """What the gateway says it charged for each call not yet settled, in place of the estimate. A record it does not have yet
        stays an estimate: this is a correction, never a wait."""
        pending, self.pending = self.pending, []
        for request_id, estimate in pending:
            charged: float | None
            try:
                record = await client.get(f"/generation?id={request_id}", cast_to=dict)
                charged = float(record["data"]["total_cost"])
            except Exception:  # noqa: BLE001  A missing record, a gateway without one, a slow one: the estimate stands.
                charged = None
            if charged is None:
                continue
            if estimate == 0.0 and self.unpriced:
                self.unpriced -= 1
            self.total += charged - estimate


async def chat(client: AsyncOpenAI, **request: Any) -> tuple[Any, str]:
    """One chat completion, and the gateway's id of the request when the client gives the headers; an empty id otherwise."""
    request.setdefault("reasoning_effort", REASONING_EFFORT)
    raw =getattr(getattr(getattr(client, "chat", None), "completions", None), "with_raw_response", None)
    if raw is None:
        return await client.chat.completions.create(**request), ""
    response = await raw.create(**request)
    return response.parse(), str(response.headers.get("X-Orca-Request-Id", "") or "")


@dataclass
class _Turn:
    """A Responses API turn, read the way a chat completion's message is. `output` is sent back whole on the next turn."""

    content: str
    tool_calls: list[Any] | None
    output: list[dict[str, Any]]

    def model_dump(self, exclude_none: bool = True) -> dict[str, Any]:
        return {"role": "assistant", "responses_output": self.output}


async def act(client: AsyncOpenAI, model: str, messages: list[Any], tools: list[dict[str, Any]]) -> tuple[Any, str]:
    """One turn of work with tools, as chat() returns it. GPT-6.1 Sol calls tools only through the Responses API, so the chat
    messages go there and its answer comes back in the shape of a completion. A client without that API takes the chat one."""
    responses = getattr(client, "responses", None)
    if responses is None:
        return await chat(client, model=model, messages=messages, tools=tools)
    response = await responses.create(
        model=model,
        instructions="\n\n".join(message["content"] for message in messages if message["role"] == "system"),
        input=_responses_input(messages),
        tools=[{**tool["function"], "type": "function", "strict": False} for tool in tools],
        reasoning={"effort": REASONING_EFFORT},
        # Nothing kept at OpenAI between turns: the reasoning comes back encrypted and is sent with the next one.
        store=False,
        include=["reasoning.encrypted_content"],
    )
    calls = [SimpleNamespace(id=item.call_id, function=SimpleNamespace(name=item.name, arguments=item.arguments)) for item in response.output if item.type == "function_call"]
    turn = _Turn(response.output_text, calls or None, [item.model_dump(exclude_none=True) for item in response.output])
    usage = response.usage
    counted = usage and SimpleNamespace(
        prompt_tokens=usage.input_tokens, completion_tokens=usage.output_tokens, prompt_tokens_details=SimpleNamespace(cached_tokens=usage.input_tokens_details.cached_tokens)
    )
    return SimpleNamespace(usage=counted, choices=[SimpleNamespace(message=turn)]), ""


def _responses_input(messages: list[Any]) -> list[dict[str, Any]]:
    """Chat messages as the Responses API's input items. The system message goes as its instructions."""
    items: list[dict[str, Any]] = []
    for message in messages:
        role, content = message["role"], message.get("content")
        if role == "assistant":
            items += message["responses_output"]
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": message["tool_call_id"], "output": content})
        elif role == "user" and isinstance(content, str):
            items.append({"role": "user", "content": content})
        elif role == "user":
            items.append({"role": "user", "content": [{"type": "input_image", "image_url": part["image_url"]["url"]} for part in content]})
    return items


SPEND = Spend()


def money(cost: float) -> str:
    """A cost as a person says it: under a cent, 4 cents, 1.20 $."""
    if cost < 0.005:
        return "under a cent"
    if cost < 1:
        return f"{cost * 100:.0f} cents"
    return f"{cost:.2f} $"


def refused(error: BaseException) -> bool:
    """Whether OpenAI refused the key: the one failure that the person must fix before anything can work."""
    return isinstance(error, AuthenticationError)


# OpenAI answers an account out of credit, or over a spend or usage limit, with a 429, the status of a rate limit, which waiting
# does not mend. Its error-codes page (read 2026-11-18) tells them apart by `code`; each can also come with the type, or an older
# code, insufficient_quota, which is taken as no credit.
NO_QUOTA = "insufficient_quota"
QUOTA_CAUSES = {
    "credit_balance_exhausted": "no_credit",
    NO_QUOTA: "no_credit",
    "organization_spend_limit_exceeded": "spend_limit",
    "project_spend_limit_exceeded": "spend_limit",
    "organization_usage_limit_exceeded": "budget",
}


def _quota_cause(error: BaseException) -> str:
    """The cause of a 429 that is about the account's money, not its pace; empty for any other error."""
    if not isinstance(error, RateLimitError):
        return ""
    code, _ = reply(error)
    if code in QUOTA_CAUSES:
        return QUOTA_CAUSES[code]
    body = getattr(error, "body", None)
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        body = body["error"]
    return "no_credit" if isinstance(body, dict) and body.get("type") == NO_QUOTA else ""


def _out_of_quota(error: BaseException) -> bool:
    return bool(_quota_cause(error))


def not_answering(error: BaseException) -> bool:
    """Whether the service failed to answer, not that it answered no: no connection, too slow, too busy, or broken on its side.
    The same request can work a moment later, so the postings in it are asked about again, not set aside. An account out of
    credit or over a limit is not one of these, though OpenAI answers it with a 429."""
    if _out_of_quota(error):
        return False
    if isinstance(error, APIConnectionError | RateLimitError):
        return True
    return isinstance(error, APIStatusError) and (error.status_code == 408 or error.status_code >= 500)


def account_problem(error: BaseException) -> bool:
    """Whether the service refused for what is about the account and not about one posting: no credit left (402, or OpenAI's
    429 insufficient_quota), no right to ask (403), or nothing there to ask (404). Asking again changes nothing, and nothing is
    set aside for it."""
    return _out_of_quota(error) or (isinstance(error, APIStatusError) and error.status_code in (402, 403, 404))


def no_credit(error: BaseException) -> bool:
    """Whether the account has no credit: OpenAI answers 429 credit_balance_exhausted, another gateway 402."""
    return _quota_cause(error) == "no_credit" or (isinstance(error, APIStatusError) and error.status_code == 402)


# The end of a gateway's message that names the request for its support and tells a person nothing.
REQUEST_ID = re.compile(r"\s*\(request id: [^)]*\)\s*$")
# A key can be in a message the service sends back; it never reaches the screen.
KEY_SHAPE = re.compile(r"sk-[A-Za-z0-9_\-]{6,}")
# How much of an unexplained refusal a person is shown.
REFUSAL_WORDS = 120
# A code the service sends is sent on to the makers only when it looks like one.
CODE_SHAPE = re.compile(r"[A-Za-z0-9_.\-]{1,60}")


def reply(error: BaseException) -> tuple[str, str]:
    """The code and the message the service gave with a refusal, read from the error's body with care: the body may be the error
    itself or wrapped in an "error" key, or missing, and the message then is the error's own text."""
    body = getattr(error, "body", None)
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        body = body["error"]
    given = body if isinstance(body, dict) else {}
    code = given.get("code")
    message = given.get("message")
    own = str(getattr(error, "message", "") or "")
    return (str(code) if code is not None else ""), (message if isinstance(message, str) else own)


# Where an OpenAI account gets credit, and where its project's limits and model access are set.
BILLING_URL = "platform.openai.com/settings/organization/billing"
LIMITS_URL = "platform.openai.com/settings/organization/limits"

# What `cause` names: every reason that is about the account or the key, so the same for the next request too.
ACCOUNT_CAUSES = frozenset({"no_credit", "free_calls_used", "key_limit", "spend_limit", "budget", "model_not_allowed", "refused_other"})


def cause(error: BaseException) -> str:
    """Why the service refused for the account, as a short name; empty when the error is not about the account. OpenAI's 429s
    about money are in QUOTA_CAUSES, a 402 is no credit; a 404 is a model the key cannot use; 403 has several causes that
    topping up does not mend, each told apart by the code the service gives or, for some, only by its message."""
    if not isinstance(error, APIStatusError):
        return ""
    code, message = reply(error)
    if quota := _quota_cause(error):
        return quota
    if error.status_code == 402:
        return "free_calls_used" if code == "free_quota_exhausted" else "no_credit"
    if error.status_code == 404:
        return "model_not_allowed" if code == "model_not_found" else ""
    if error.status_code != 403:
        return ""
    if code == "pre_consume_token_quota_failed" or "token quota is not enough" in message:
        return "key_limit"
    if "token cycle spend limit reached" in message:
        return "spend_limit"
    if code == "budget_exceeded":
        return "budget"
    if "has no access to model" in message:
        return "model_not_allowed"
    return "refused_other"


def error_name(error: BaseException) -> str:
    """An error as the makers are told of it: its class, then the status and the service's code when it has them, such as
    "APIStatusError 402 insufficient_user_quota". The class alone does not say which 4xx it was."""
    name = type(error).__name__
    if not isinstance(error, APIStatusError):
        return name
    code, _ = reply(error)
    return " ".join([name, str(error.status_code), *([code] if CODE_SHAPE.fullmatch(code) else [])])


def refusal_in_words(error: BaseException) -> str:
    """The service's own message for a refusal nothing above explains, without its request id, cut to a few words."""
    _, message = reply(error)
    message = KEY_SHAPE.sub("sk-…", REQUEST_ID.sub("", message)).strip()
    if message.startswith("Error code:"):
        # The error's own text, not the service's: it says nothing a person can use.
        return ""
    return message if len(message) <= REFUSAL_WORDS else message[: REFUSAL_WORDS - 1].rstrip() + "…"


def known(error: BaseException) -> bool:
    """Whether `explain` has words for this kind of error. One it has none for is unexpected, and goes to the error log."""
    return isinstance(error, APIConnectionError | RateLimitError | InternalServerError | BadRequestError) or account_problem(error)


def explain(error: Exception, step: str = "press → to go on.") -> str:
    """What a failed model call says in plain words and what to do about it; `step` is what to press once the account is mended,
    for the place the call failed in. The error's own text stays out, but for a refusal nothing explains: the key can be in it."""
    why = cause(error)
    if why == "no_credit":
        return f"Your OpenAI account has no credit left. Add some at {BILLING_URL}, then {step}"
    if why == "free_calls_used":
        return f"The free calls for today are used up. Add credit at {BILLING_URL}, then {step}"
    after = f" Then {step}"
    if why == "key_limit":
        return f"This key's own limit is used up. Raise it, or make a new key, at {LIMITS_URL}.{after}"
    if why == "spend_limit":
        return f"This key reached its spending limit for now. Raise it at {LIMITS_URL}, or wait until it resets.{after}"
    if why == "budget":
        return f"Your OpenAI budget for this month is used up. Raise it at {LIMITS_URL}.{after}"
    if why == "model_not_allowed":
        return f"This key is not allowed to use the models AIHawk needs. Check the project's model access at {LIMITS_URL}.{after}"
    if why == "refused_other":
        said = refusal_in_words(error)
        return f"OpenAI refused this key for the request{': ' + said.rstrip('.') if said else ''}.{after}"
    if account_problem(error):
        return f"The AI service refused the request. Check your OpenAI account at platform.openai.com, then {step}"
    if isinstance(error, APITimeoutError):
        return "The AI service did not answer in time. Try again in a moment."
    if isinstance(error, APIConnectionError):
        return "Cannot reach the AI service. Check your internet connection."
    if isinstance(error, RateLimitError):
        return "The AI service is busy. Wait a minute."
    if isinstance(error, InternalServerError):
        return "The AI service has a problem on its side. Try again in a few minutes."
    if isinstance(error, BadRequestError):
        return "The AI service did not accept the request."
    return "Something went wrong. Try again in a moment."


def make_client(config: Config) -> AsyncOpenAI:
    key = config.api_key()
    if not key:
        raise ConfigError(f"no model key: set {KEY_ENV} (create one at {KEY_URL})")
    # The SDK retries 429s and 5xx itself and honours Retry-After.
    return AsyncOpenAI(base_url=DEFAULT_BASE_URL, api_key=key, max_retries=5, timeout=TIMEOUT)


async def structured(client: AsyncOpenAI, model: str, system: str, user: str, name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """One answer that follows `schema`. A reply that is not that JSON raises: nothing is guessed from it."""
    response, request_id = await chat(
        client,
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}},
    )
    SPEND.note(model, getattr(response, "usage", None), request_id)
    content = response.choices[0].message.content
    if not content:
        raise ValueError(f"the model gave no {name}")
    return json.loads(content)
