import asyncio
import random
import time
from typing import Any, cast, overload

import tiktoken
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAI,
    RateLimitError,
)
from tqdm.asyncio import tqdm as async_tqdm

from .type_defs import LLMClientProtocol, ResponseFormatType


class _RateLimiter:
    """Sliding-window rate limiter for async use."""

    def __init__(self, max_per_minute: int):
        self._rpm = max_per_minute
        self._lock = asyncio.Lock()
        self._timestamps: list[float] = []

    async def acquire(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            self._timestamps = [t for t in self._timestamps if now - t < 60.0]
            if len(self._timestamps) >= self._rpm:
                sleep_for = 60.0 - (now - self._timestamps[0])
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                now = asyncio.get_event_loop().time()
                self._timestamps = [t for t in self._timestamps if now - t < 60.0]
            self._timestamps.append(asyncio.get_event_loop().time())


def _should_retry(exc: Exception) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIError) and exc.status_code is not None and exc.status_code >= 500:
        return True
    return False


async def _retry_async(coro_fn, max_retries: int, timeout: float | None):
    for attempt in range(max_retries + 1):
        try:
            coro = coro_fn()
            if timeout is not None:
                return await asyncio.wait_for(coro, timeout=timeout)
            return await coro
        except Exception as exc:
            if attempt == max_retries or not _should_retry(exc):
                raise
            # Longer base delay for rate limit errors
            base = 10.0 if isinstance(exc, RateLimitError) else 1.0
            delay = base * (2**attempt) + random.uniform(0, 1)
            await asyncio.sleep(delay)


def _retry_sync(call_fn, max_retries: int):
    for attempt in range(max_retries + 1):
        try:
            return call_fn()
        except Exception as exc:
            if attempt == max_retries or not _should_retry(exc):
                raise
            base = 10.0 if isinstance(exc, RateLimitError) else 1.0
            delay = base * (2**attempt) + random.uniform(0, 1)
            time.sleep(delay)


class OpenAIClient(LLMClientProtocol):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.4-mini",
        embedding_model: str = "text-embedding-3-small",
        max_tokens: int = 1024,
        base_url: str = "https://api.openai.com/v1",
        **kwargs,
    ):
        self.model = model
        self.embedding_model = embedding_model
        self.max_tokens = max_tokens
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._tokenizer: tiktoken.Encoding | None = None

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            try:
                self._tokenizer = tiktoken.encoding_for_model(self.model)
            except KeyError:
                self._tokenizer = tiktoken.get_encoding("o200k_base")
        return len(self._tokenizer.encode(text))

    def embed(self, text: str | list[str], **kwargs) -> list[list[float]]:
        texts = [text] if isinstance(text, str) else text
        response = self._client.embeddings.create(model=self.embedding_model, input=texts, **kwargs)
        return [item.embedding for item in response.data]

    def _build_messages(
        self, system_prompt: str, user_prompt: str
    ) -> list[Any]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @overload
    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: None = None,
        timeout: float | None = None,
        max_retries: int = 3,
        **kwargs,
    ) -> str: ...

    @overload
    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: type[ResponseFormatType],
        timeout: float | None = None,
        max_retries: int = 3,
        **kwargs,
    ) -> ResponseFormatType: ...

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: type[ResponseFormatType] | None = None,
        timeout: float | None = None,
        max_retries: int = 3,
        **kwargs,
    ) -> str | ResponseFormatType:
        messages = self._build_messages(system_prompt, user_prompt)

        if response_format is None:
            def _call():
                return self._client.responses.create(
                    model=self.model,
                    input=messages,
                    max_output_tokens=self.max_tokens,
                    timeout=timeout,
                    **kwargs
                )
        else:
            def _call():
                return self._client.responses.parse(
                    model=self.model,
                    input=messages,
                    max_output_tokens=self.max_tokens,
                    timeout=timeout,
                    text_format=response_format,
                    **kwargs
                )

        resp = _retry_sync(_call, max_retries)
        content = resp.output_text or ""
        if response_format is None:
            return content
        return response_format.model_validate_json(content)

    @overload
    async def call_async(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: None = None,
        timeout: float | None = None,
        max_retries: int = 3,
        **kwargs,
    ) -> str: ...

    @overload
    async def call_async(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: type[ResponseFormatType],
        timeout: float | None = None,
        max_retries: int = 3,
        **kwargs,
    ) -> ResponseFormatType: ...

    async def call_async(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: type[ResponseFormatType] | None = None,
        timeout: float | None = None,
        max_retries: int = 3,
        **kwargs,
    ) -> str | ResponseFormatType:
        messages = self._build_messages(system_prompt, user_prompt)

        if response_format is None:
            def _coro():
                return self._async_client.responses.create(
                    model=self.model,
                    input=messages,
                    max_output_tokens=self.max_tokens,
                    timeout=timeout,
                    **kwargs,
                )
        else:
            def _coro():
                return self._async_client.responses.parse(
                    model=self.model,
                    input=messages,
                    max_output_tokens=self.max_tokens,
                    timeout=timeout,
                    text_format=response_format,
                    **kwargs,
                )

        resp = await _retry_async(_coro, max_retries, timeout)
        content = resp.output_text or ""
        if response_format is None:
            return content
        return response_format.model_validate_json(content)

    @overload
    async def call_many(
        self,
        system_prompt: str | list[str],
        user_prompt: str | list[str],
        response_format: None = None,
        max_requests_per_minute: int = 100,
        progress: bool = True,
        timeout: float | None = None,
        max_retries: int = 3,
        **kwargs,
    ) -> list[str]: ...

    @overload
    async def call_many(
        self,
        system_prompt: str | list[str],
        user_prompt: str | list[str],
        response_format: type[ResponseFormatType],
        max_requests_per_minute: int = 100,
        progress: bool = True,
        timeout: float | None = None,
        max_retries: int = 3,
        **kwargs,
    ) -> list[ResponseFormatType]: ...

    async def call_many(
        self,
        system_prompt: str | list[str],
        user_prompt: str | list[str],
        response_format: type[ResponseFormatType] | None = None,
        max_requests_per_minute: int = 100,
        progress: bool = True,
        timeout: float | None = None,
        max_retries: int = 3,
        **kwargs,
    ) -> list[str] | list[ResponseFormatType]:
        prompts = [user_prompt] if isinstance(user_prompt, str) else user_prompt
        systems = (
            [system_prompt] * len(prompts) if isinstance(system_prompt, str) else system_prompt
        )
        limiter = _RateLimiter(max_requests_per_minute)

        async def _bounded_call(sys: str, usr: str):
            await limiter.acquire()
            return await self.call_async(
                sys, usr, response_format=response_format,
                timeout=timeout, max_retries=max_retries, **kwargs
            )

        coroutines = [_bounded_call(sys, usr) for sys, usr in zip(systems, prompts)]
        results: list[str|ResponseFormatType]
        if progress:
            results  = await async_tqdm.gather(*coroutines, ncols=60)
        else:
            results = await asyncio.gather(*coroutines)
        return cast(list[Any], results)

    def send_messages(
        self,
        messages: list[dict],
        timeout: float | None = None,
        max_retries: int = 3,
        **kwargs,
    ) -> str:
        def _call():
            return self._client.chat.completions.create(
                model=self.model,
                messages=cast(list[Any], messages),
                max_completion_tokens=self.max_tokens,
                timeout=timeout,
                **kwargs,
            )

        resp = _retry_sync(_call, max_retries)
        return resp.choices[0].message.content or ""

    async def send_messages_async(
        self,
        messages: list[dict],
        timeout: float | None = None,
        max_retries: int = 3,
        **kwargs,
    ) -> str:
        def _coro():
            return self._async_client.chat.completions.create(
                model=self.model,
                messages=cast(list[Any], messages),
                max_completion_tokens=self.max_completion_tokens,
                **kwargs,
            )

        resp = await _retry_async(_coro, max_retries, timeout)
        return resp.choices[0].message.content or ""
