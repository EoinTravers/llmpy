# %% [markdown]
# #dsllmpy Quickstart

# %% [markdown]
# ## OpenAI

# %%
import os
from dsllmpy import OpenAIClient

openai_key = os.environ.get("OPENAI_API_KEY")
if not openai_key:
    print("OPENAI_API_KEY not set — skipping OpenAI examples")
else:
    client = OpenAIClient(api_key=openai_key, model="gpt-4o-mini")
    response = client.call("You are helpful.", "What is the capital of France?")
    print(response)

# %% [markdown]
# ### Structured output

# %%
from pydantic import BaseModel


class Answer(BaseModel):
    answer: str
    confidence: float


if openai_key:
    result = client.call("You are helpful.", "What is 2+2?", response_format=Answer)
    print(result.answer, result.confidence)

# %% [markdown]
# ## Ollama (local models)

# %%
ollama_client = OpenAIClient(
    api_key="ollama",
    model="smollm:135M",
    base_url="http://localhost:11434/v1",
)

response = ollama_client.call("You are helpful.", "Write a short poem about the colour blue.")
print(response)

# %% [markdown]
# ## Async and batch

# %%
import asyncio


async def main():
    response = await ollama_client.call_async("You are helpful.", "Say hello.")
    print(response)

    responses = await ollama_client.call_many(
        "You are helpful.", ["Say hi.", "Say bye.", "Say thanks."]
    )
    for r in responses:
        print(r)


asyncio.run(main())
