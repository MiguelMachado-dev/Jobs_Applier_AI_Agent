"""A turn with tools through the Responses API: what is sent, what is sent back, and how it reads as a chat message. No network."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from openai import AsyncOpenAI

from aihawk.config import DEFAULT_BASE_URL, REASONING_EFFORT
from aihawk.llm import act

TOOL = {"type": "function", "function": {"name": "browser_snapshot", "description": "Read the page.", "parameters": {"type": "object", "properties": {}}}}
REASONING = {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "sealed"}
CALL = {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "browser_snapshot", "arguments": "{}", "status": "completed"}


def reply(*output: dict[str, Any]) -> httpx.Response:
    usage = {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 40}, "output_tokens": 7, "output_tokens_details": {"reasoning_tokens": 3}, "total_tokens": 107}
    body = {"id": "r", "object": "response", "created_at": 0, "model": "m", "output": list(output), "parallel_tool_calls": True, "tool_choice": "auto", "tools": [], "usage": usage}
    return httpx.Response(200, json=body)


def client_answering(seen: list[dict[str, Any]], *replies: httpx.Response) -> AsyncOpenAI:
    answers = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == DEFAULT_BASE_URL + "/responses"
        seen.append(json.loads(request.content))
        return answers.pop(0)

    return AsyncOpenAI(base_url=DEFAULT_BASE_URL, api_key="k", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def test_a_turn_is_sent_as_the_responses_api_wants_and_reads_as_a_chat_message() -> None:
    seen: list[dict[str, Any]] = []
    messages = [{"role": "system", "content": "Rules."}, {"role": "user", "content": "Fill it."}]
    response, request_id = asyncio.run(act(client_answering(seen, reply(REASONING, CALL)), "m", messages, [TOOL]))
    (sent,) = seen
    assert sent["instructions"] == "Rules." and sent["input"] == [{"role": "user", "content": "Fill it."}]
    assert sent["tools"] == [{"type": "function", "name": "browser_snapshot", "description": "Read the page.", "parameters": {"type": "object", "properties": {}}, "strict": False}]
    assert sent["reasoning"] == {"effort": REASONING_EFFORT} and sent["store"] is False and sent["include"] == ["reasoning.encrypted_content"]
    (call,) = response.choices[0].message.tool_calls
    assert (call.id, call.function.name, call.function.arguments, request_id) == ("call_1", "browser_snapshot", "{}", "")
    assert (response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.prompt_tokens_details.cached_tokens) == (100, 7, 40)


def test_the_next_turn_carries_the_reasoning_the_calls_their_answers_and_the_picture() -> None:
    seen: list[dict[str, Any]] = []
    client = client_answering(seen, reply(REASONING, CALL), reply({"id": "m", "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "Done.", "annotations": []}]}))

    async def run() -> Any:
        messages: list[Any] = [{"role": "system", "content": "Rules."}, {"role": "user", "content": "Fill it."}]
        first, _ = await act(client, "m", messages, [TOOL])
        messages.append(first.choices[0].message.model_dump(exclude_none=True))
        messages.append({"role": "tool", "tool_call_id": "call_1", "content": "A form."})
        messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]})
        return await act(client, "m", messages, [TOOL])

    second, _ = asyncio.run(run())
    items = seen[1]["input"]
    assert items[1] == REASONING and items[2]["call_id"] == "call_1" and items[2]["type"] == "function_call"
    assert items[3] == {"type": "function_call_output", "call_id": "call_1", "output": "A form."}
    assert items[4] == {"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA"}]}
    assert second.choices[0].message.content == "Done." and second.choices[0].message.tool_calls is None
