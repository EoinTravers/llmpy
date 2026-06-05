from typing import Protocol, TypeVar, overload

from pydantic import BaseModel

ResponseFormatType = TypeVar("ResponseFormatType", bound=BaseModel)


class LLMClientProtocol(Protocol):
    def __init__(
        self,
        api_key: str,
        model: str,
        embedding_model: str,
        max_tokens: int,
        base_url: str,
        **kwargs,
    ):
        """Initialize the LLM client."""
        ...

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in text."""
        ...

    def embed(self, text: str | list[str], **kwargs) -> list[list[float]]:
        """Embed text using the LLM's embedding model."""
        ...

    @overload  # Variant with no response format
    def call(
        self, system_prompt: str, user_prompt: str, response_format: None = None, **kwargs
    ) -> str:
        """Call the LLM with a system prompt and a user prompt."""
        ...

    @overload  # Variant with a response format
    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: type[ResponseFormatType],
        **kwargs,
    ) -> ResponseFormatType:
        """Call the LLM with a system prompt and a user prompt."""
        ...

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: type[ResponseFormatType] | None = None,
        **kwargs,
    ) -> str | ResponseFormatType:
        """Call the LLM with a system prompt and a user prompt."""
        ...

    @overload  # Variant with no response format
    async def call_async(
        self, system_prompt: str, user_prompt: str, response_format: None = None, **kwargs
    ) -> str:
        """Call the LLM with a system prompt and a user prompt asynchronously."""
        ...

    @overload  # Variant with a response format
    async def call_async(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: type[ResponseFormatType],
        **kwargs,
    ) -> ResponseFormatType:
        """Call the LLM with a system prompt and a user prompt asynchronously."""
        ...

    async def call_async(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: type[ResponseFormatType] | None = None,
        **kwargs,
    ) -> str | ResponseFormatType:
        """Call the LLM with a system prompt and a user prompt asynchronously."""
        ...

    @overload  # Variant with no response format
    async def call_batch(
        self,
        system_prompt: str | list[str],
        user_prompt: str | list[str],
        response_format: None = None,
        max_requests_per_minute: int = 100,
        **kwargs,
    ) -> list[str]:
        """Call the LLM with a system prompt and a user prompt in batch mode."""
        ...

    @overload  # Variant with a response format
    async def call_batch(
        self,
        system_prompt: str | list[str],
        user_prompt: str | list[str],
        response_format: type[ResponseFormatType],
        max_requests_per_minute: int = 100,
        **kwargs,
    ) -> list[ResponseFormatType]:
        """Call the LLM with a system prompt and a user prompt in batch mode."""
        ...

    async def call_batch(
        self,
        system_prompt: str | list[str],
        user_prompt: str | list[str],
        response_format: type[ResponseFormatType] | None = None,
        max_requests_per_minute: int = 100,
        **kwargs,
    ) -> list[str] | list[ResponseFormatType]:
        """Call the LLM with a system prompt and a user prompt in batch mode."""
        ...

    def send_messages(self, messages: list[dict], **kwargs) -> str:
        """Send a list of messages to the LLM."""
        ...

    async def send_messages_async(self, messages: list[dict], **kwargs) -> str:
        """Send a list of messages to the LLM asynchronously."""
        ...
