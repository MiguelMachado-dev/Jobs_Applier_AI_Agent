"""The one model gateway: OpenAI's chat API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from openai import (
    APIConnectionError,
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
# The SDK's own default is ten minutes a try, and a model that hangs would hold a judgment that long.
TIMEOUT = 180.0


class Spend:
    """What the model calls cost, in dollars: each call's tokens at the gateway's own price for its model, added up.

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
        estimate = 0.0
        if usage is None:
            # A client that says nothing about tokens: nothing to estimate, and nothing to call unpriced.
            if request_id:
                self.pending.append((request_id, 0.0))
            return 0.0
        if (price := self.prices.get(model)) is not None:
            prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion = int(getattr(usage, "completion_tokens", 0) or 0)
            details = getattr(usage, "prompt_tokens_details", None)
            cached = int(getattr(details, "cached_tokens", 0) or 0)
            estimate = (prompt - cached) * price[0] + cached * price[2] + completion * price[1]
            self.total += estimate
        else:
            self.unpriced += 1
        if request_id:
            self.pending.append((request_id, estimate))
        return estimate

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
    raw = getattr(getattr(getattr(client, "chat", None), "completions", None), "with_raw_response", None)
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


def explain(error: Exception) -> str:
    """What a failed model call says in plain words. The detail stays after it, and the key is never in it."""
    if isinstance(error, APITimeoutError):
        return f"OpenAI did not answer in time. Try again in a moment. ({error})"
    if isinstance(error, APIConnectionError):
        return f"OpenAI cannot be reached: check your internet connection and try again. ({error})"
    if isinstance(error, RateLimitError):
        return f"OpenAI is asking AIHawk to slow down: wait a few minutes and try again. ({error})"
    if isinstance(error, InternalServerError):
        return f"OpenAI failed on its side: try again in a few minutes. ({error})"
    if isinstance(error, BadRequestError):
        return f"OpenAI rejected the request. ({error})"
    return f"Something failed: {error}"


def make_client(config: Config) -> AsyncOpenAI:
    key = config.api_key()
    if not key:
        raise ConfigError(f"no model key: set {KEY_ENV} (create one at {KEY_URL})")
    # The SDK retries 429s and 5xx itself and honours Retry-After, which is all OpenAI sends about its limits.
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
