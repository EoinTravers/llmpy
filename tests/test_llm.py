"""Tests for LLMClientProtocol — verifies the contract any implementation must satisfy."""

import asyncio

import pytest
from pydantic import BaseModel


class _SampleResponse(BaseModel):
    answer: str
    confidence: float


class _MockLLMClient:
    """Minimal conforming implementation of LLMClientProtocol used as the test subject."""

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed(self, text: str | list[str], **kwargs) -> list[list[float]]:
        texts = [text] if isinstance(text, str) else text
        return [[float(i) * 0.1 for i in range(3)] for _ in texts]

    def call(self, system_prompt, user_prompt, response_format=None, **kwargs):
        if response_format is None:
            return f"response:{user_prompt}"
        return response_format(answer=user_prompt, confidence=0.9)

    async def call_async(self, system_prompt, user_prompt, response_format=None, **kwargs):
        if response_format is None:
            return f"async:{user_prompt}"
        return response_format(answer=user_prompt, confidence=0.8)

    async def call_batch(
        self,
        system_prompt,
        user_prompt,
        response_format=None,
        max_requests_per_minute=100,
        **kwargs,
    ):
        prompts = [user_prompt] if isinstance(user_prompt, str) else user_prompt

        if response_format is None:
            return [f"batch:{p}" for p in prompts]
        return [response_format(answer=p, confidence=0.7) for p in prompts]

    def send_messages(self, messages: list[dict], **kwargs) -> str:
        return f"messages:{len(messages)}"

    async def send_messages_async(self, messages: list[dict], **kwargs) -> str:
        return f"async_messages:{len(messages)}"


@pytest.fixture
def client() -> _MockLLMClient:
    return _MockLLMClient()


# --- count_tokens ---

def test_count_tokens_returns_int(client):
    result = client.count_tokens("hello world")
    assert isinstance(result, int)


def test_count_tokens_empty_string(client):
    result = client.count_tokens("")
    assert isinstance(result, int)


def test_count_tokens_longer_text_is_more(client):
    short = client.count_tokens("hello")
    long = client.count_tokens("hello world foo bar baz")
    assert long > short


# --- embed ---

def test_embed_single_string(client):
    result = client.embed("hello")
    assert isinstance(result, list)
    assert len(result) == 1
    assert all(isinstance(v, float) for v in result[0])


def test_embed_list_of_strings(client):
    result = client.embed(["hello", "world", "foo"])
    assert len(result) == 3
    assert all(isinstance(row, list) for row in result)
    assert all(isinstance(v, float) for row in result for v in row)


# --- call (sync) ---

def test_call_no_format_returns_str(client):
    result = client.call("sys", "user")
    assert isinstance(result, str)


def test_call_with_format_returns_model(client):
    result = client.call("sys", "what is 2+2?", response_format=_SampleResponse)
    assert isinstance(result, _SampleResponse)


def test_call_passes_user_prompt(client):
    result = client.call("sys", "ping", response_format=None)
    assert "ping" in result


def test_call_with_format_fields(client):
    result = client.call("sys", "hello", response_format=_SampleResponse)
    assert isinstance(result.answer, str)
    assert 0.0 <= result.confidence <= 1.0


# --- call_async ---

def test_call_async_no_format_returns_str(client):
    result = asyncio.run(client.call_async("sys", "user"))
    assert isinstance(result, str)


def test_call_async_with_format_returns_model(client):
    result = asyncio.run(client.call_async("sys", "user", response_format=_SampleResponse))
    assert isinstance(result, _SampleResponse)


# --- call_batch ---

def test_call_batch_returns_list_of_str(client):
    result = asyncio.run(client.call_batch("sys", ["a", "b", "c"]))
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(r, str) for r in result)


def test_call_batch_with_format_returns_list_of_models(client):
    result = asyncio.run(
        client.call_batch("sys", ["a", "b"], response_format=_SampleResponse)
    )
    assert len(result) == 2
    assert all(isinstance(r, _SampleResponse) for r in result)


def test_call_batch_per_prompt_systems(client):
    result = asyncio.run(
        client.call_batch(["sys1", "sys2"], ["a", "b"])
    )
    assert len(result) == 2


def test_call_batch_length_matches_prompts(client):
    prompts = ["x", "y", "z", "w"]
    result = asyncio.run(client.call_batch("sys", prompts))
    assert len(result) == len(prompts)


# --- send_messages ---

def test_send_messages_returns_str(client):
    messages = [{"role": "user", "content": "hello"}]
    result = client.send_messages(messages)
    assert isinstance(result, str)


def test_send_messages_multi_turn(client):
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "bye"},
    ]
    result = client.send_messages(messages)
    assert isinstance(result, str)


# --- send_messages_async ---

def test_send_messages_async_returns_str(client):
    messages = [{"role": "user", "content": "hello"}]
    result = asyncio.run(client.send_messages_async(messages))
    assert isinstance(result, str)


# --- protocol conformance ---

def test_mock_satisfies_protocol():
    """Verify _MockLLMClient is a structural subtype of LLMClientProtocol at runtime.

    LLMClientProtocol is not @runtime_checkable, so we verify by checking that
    all required method names are present on the implementation.
    """
    required = {
        "count_tokens", "embed", "call", "call_async",
        "call_batch", "send_messages", "send_messages_async",
    }
    assert required.issubset(dir(_MockLLMClient))
