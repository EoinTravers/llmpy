# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                          # install dependencies
uv run pytest                    # run all tests
uv run pytest tests/test_llm.py::test_call_no_format_returns_str  # run a single test
uv run ruff check src tests      # lint
uv run ty src                  # type check
```

## Architecture

This is a library (`dsllmpy`) defining a standard interface for LLM clients.

**`src/llm.py`** — the central artifact: `LLMClientProtocol`, a `typing.Protocol` that any LLM client implementation must satisfy. It is imported as `from llm import LLMClientProtocol` (top-level `src/` module, not inside the `dsllmpy` package).

The protocol defines:
- `call` / `call_async` / `call_many` — overloaded so that passing `response_format=None` returns `str`, passing a Pydantic `BaseModel` subclass returns an instance of that type.
- `call_many` — accepts either a single shared `system_prompt: str` or a per-request `list[str]`, paired with a `list[str]` of user prompts.
- `send_messages` / `send_messages_async` — raw multi-turn message list interface (`list[dict]`).
- `embed` — returns `list[list[float]]` for one or more input strings.
- `count_tokens` — returns `int`.

**`src/dsllmpy/`** — the installable package; currently a stub. New implementations of `LLMClientProtocol` should live here and be exported from `__init__.py`.

**`tests/test_llm.py`** — validates the protocol contract using `_MockLLMClient`, a minimal conforming implementation. New protocol methods need corresponding tests here.
