"""Unit tests for OpenAIClient using mocked API responses."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from openai import APIConnectionError, APIError, RateLimitError
from pydantic import BaseModel

from llmpy.openai_client import OpenAIClient, _should_retry, _retry_sync, _retry_async, _RateLimiter


@pytest.fixture
def client():
    return OpenAIClient(api_key="test", model="gpt-4o-mini", max_tokens=64)


def _make_completion(content: str):
    msg = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


def _make_embedding(vectors: list[list[float]]):
    items = [SimpleNamespace(embedding=v) for v in vectors]
    return SimpleNamespace(data=items)


# --- count_tokens ---

def test_count_tokens_returns_int(client):
    assert isinstance(client.count_tokens("hello world"), int)


def test_count_tokens_two_words(client):
    assert client.count_tokens("hello world") == 2


def test_count_tokens_longer_text_is_more(client):
    assert client.count_tokens("hello world foo bar baz qux") > client.count_tokens("hi")


# --- call ---

def test_call_returns_str(client):
    with patch.object(client._client.chat.completions, "create",
                      return_value=_make_completion("hello")):
        result = client.call("sys", "usr")
    assert isinstance(result, str)
    assert result == "hello"


def test_call_response_format_returns_model(client):
    class _G(BaseModel):
        word: str

    with patch.object(client._client.chat.completions, "create",
                      return_value=_make_completion('{"word": "hello"}')):
        result = client.call("sys", "usr", response_format=_G)
    assert isinstance(result, _G)
    assert result.word == "hello"


def test_call_passes_json_object_format_when_response_format_set(client):
    class _G(BaseModel):
        word: str

    with patch.object(client._client.chat.completions, "create",
                      return_value=_make_completion('{"word": "hi"}')) as mock:
        client.call("sys", "usr", response_format=_G)
    _, kwargs = mock.call_args
    assert kwargs.get("response_format") == {"type": "json_object"}


def test_call_no_response_format_type_sent(client):
    with patch.object(client._client.chat.completions, "create",
                      return_value=_make_completion("hi")) as mock:
        client.call("sys", "usr")
    _, kwargs = mock.call_args
    assert "response_format" not in kwargs


# --- call_async ---

def test_call_async_returns_str(client):
    async def _fake(**kwargs):
        return _make_completion("hello")

    with patch.object(client._async_client.chat.completions, "create", side_effect=_fake):
        result = asyncio.run(client.call_async("sys", "usr"))
    assert isinstance(result, str)
    assert result == "hello"


def test_call_async_response_format_returns_model(client):
    class _G(BaseModel):
        word: str

    async def _fake(**kwargs):
        return _make_completion('{"word": "hi"}')

    with patch.object(client._async_client.chat.completions, "create", side_effect=_fake):
        result = asyncio.run(client.call_async("sys", "usr", response_format=_G))
    assert isinstance(result, _G)
    assert result.word == "hi"


# --- call_batch ---

def test_call_batch_returns_list_of_str(client):
    async def _fake(**kwargs):
        return _make_completion("ok")

    with patch.object(client._async_client.chat.completions, "create", side_effect=_fake):
        results = asyncio.run(client.call_batch("sys", ["p1", "p2"], progress=False))
    assert results == ["ok", "ok"]


def test_call_batch_length_matches_prompts(client):
    async def _fake(**kwargs):
        return _make_completion("x")

    with patch.object(client._async_client.chat.completions, "create", side_effect=_fake):
        results = asyncio.run(client.call_batch("sys", ["a", "b", "c"], progress=False))
    assert len(results) == 3


def test_call_batch_per_system_prompt(client):
    async def _fake(**kwargs):
        return _make_completion("x")

    with patch.object(client._async_client.chat.completions, "create", side_effect=_fake):
        results = asyncio.run(
            client.call_batch(["sys1", "sys2"], ["p1", "p2"], progress=False)
        )
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)


def test_call_batch_response_format_returns_list_of_models(client):
    class _G(BaseModel):
        word: str

    responses = ['{"word": "hello"}', '{"word": "bye"}']
    idx = 0

    async def _fake(**kwargs):
        nonlocal idx
        r = _make_completion(responses[idx % len(responses)])
        idx += 1
        return r

    with patch.object(client._async_client.chat.completions, "create", side_effect=_fake):
        results = asyncio.run(
            client.call_batch("sys", ["p1", "p2"], response_format=_G, progress=False)
        )
    assert all(isinstance(r, _G) for r in results)
    assert [r.word for r in results] == ["hello", "bye"]


# --- send_messages ---

def test_send_messages_returns_str(client):
    with patch.object(client._client.chat.completions, "create",
                      return_value=_make_completion("hello")):
        result = client.send_messages([{"role": "user", "content": "hi"}])
    assert isinstance(result, str)
    assert result == "hello"


def test_send_messages_multi_turn(client):
    messages = [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Hi Alice!"},
        {"role": "user", "content": "What is my name?"},
    ]
    with patch.object(client._client.chat.completions, "create",
                      return_value=_make_completion("Alice")) as mock:
        result = client.send_messages(messages)
    assert result == "Alice"
    assert mock.call_args[1]["messages"] == messages


def test_send_messages_async_returns_str(client):
    async def _fake(**kwargs):
        return _make_completion("hello")

    with patch.object(client._async_client.chat.completions, "create", side_effect=_fake):
        result = asyncio.run(client.send_messages_async([{"role": "user", "content": "hi"}]))
    assert result == "hello"


# --- count_tokens fallback encoding ---

def test_count_tokens_unknown_model_falls_back():
    client_unk = OpenAIClient(api_key="test", model="unknown-model-xyz")
    assert isinstance(client_unk.count_tokens("hello world"), int)


# --- _should_retry ---

def test_should_retry_connection_error():
    assert _should_retry(APIConnectionError(request=MagicMock())) is True


def test_should_retry_rate_limit_error():
    assert _should_retry(RateLimitError(message="rate limit", response=MagicMock(), body={})) is True


def test_should_retry_api_error_5xx():
    err = APIError(message="server error", request=MagicMock(), body={})
    err.status_code = 503
    assert _should_retry(err) is True


def test_should_retry_api_error_4xx_is_false():
    err = APIError(message="bad request", request=MagicMock(), body={})
    err.status_code = 400
    assert _should_retry(err) is False


def test_should_retry_generic_exception_is_false():
    assert _should_retry(ValueError("nope")) is False


# --- _retry_sync ---

def test_retry_sync_retries_on_retryable_error():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise APIConnectionError(request=MagicMock())
        return "ok"

    with patch("llmpy.openai_client.time.sleep"):
        result = _retry_sync(flaky, max_retries=3)
    assert result == "ok"
    assert len(calls) == 3


def test_retry_sync_raises_after_max_retries():
    def always_fail():
        raise APIConnectionError(request=MagicMock())

    with patch("llmpy.openai_client.time.sleep"):
        with pytest.raises(APIConnectionError):
            _retry_sync(always_fail, max_retries=2)


def test_retry_sync_does_not_retry_non_retryable():
    calls = []

    def fail_once():
        calls.append(1)
        raise ValueError("nope")

    with pytest.raises(ValueError):
        _retry_sync(fail_once, max_retries=3)
    assert len(calls) == 1


# --- _retry_async ---

def test_retry_async_retries_on_retryable_error():
    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise APIConnectionError(request=MagicMock())
        return "ok"

    with patch("llmpy.openai_client.asyncio.sleep", new_callable=AsyncMock):
        result = asyncio.run(_retry_async(flaky, max_retries=3, timeout=None))
    assert result == "ok"
    assert len(calls) == 2


def test_retry_async_raises_after_max_retries():
    async def always_fail():
        raise APIConnectionError(request=MagicMock())

    with patch("llmpy.openai_client.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(APIConnectionError):
            asyncio.run(_retry_async(always_fail, max_retries=1, timeout=None))


# --- _RateLimiter ---

def test_rate_limiter_allows_requests_under_limit():
    async def _run():
        limiter = _RateLimiter(10)
        for _ in range(5):
            await limiter.acquire()

    asyncio.run(_run())


def test_rate_limiter_sleeps_when_limit_reached():
    slept = []

    async def _run():
        with patch("llmpy.openai_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            limiter = _RateLimiter(2)
            # Pre-fill timestamps so the next acquire hits the limit
            import asyncio as _asyncio
            now = _asyncio.get_event_loop().time()
            limiter._timestamps = [now, now]
            await limiter.acquire()
            slept.append(mock_sleep.call_count)

    asyncio.run(_run())
    assert slept[0] >= 1


# --- embed ---

def test_embed_single_string_returns_nested_list(client):
    with patch.object(client._client.embeddings, "create",
                      return_value=_make_embedding([[0.1, 0.2, 0.3]])):
        result = client.embed("hello")
    assert result == [[0.1, 0.2, 0.3]]


def test_embed_list_returns_one_vector_per_input(client):
    vecs = [[0.1, 0.2], [0.3, 0.4]]
    with patch.object(client._client.embeddings, "create",
                      return_value=_make_embedding(vecs)):
        result = client.embed(["a", "b"])
    assert result == vecs
