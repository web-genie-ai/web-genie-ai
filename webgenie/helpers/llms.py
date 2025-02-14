import bittensor as bt

from openai import AsyncOpenAI

from webgenie.constants import (
    LLM_MODEL_ID, 
    LLM_API_KEY, 
    LLM_MODEL_URL
)

if not LLM_API_KEY or not LLM_MODEL_URL or not LLM_MODEL_ID:
    raise Exception("LLM_API_KEY, LLM_MODEL_URL, and LLM_MODEL_ID must be set")

client = AsyncOpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_MODEL_URL,
)


SEED = 42


def set_seed(seed):
    global SEED
    SEED = seed


async def openai_call(messages, response_format, deterministic=True, retries=3):
    for _ in range(retries):
        try:
            if deterministic:
                completion = await client.beta.chat.completions.parse(
                    model=LLM_MODEL_ID,
                    messages= messages,
                    response_format=response_format,
                    temperature=0,
                    seed=SEED,
                )
            else:
                completion = await client.beta.chat.completions.parse(
                    model=LLM_MODEL_ID,
                    messages= messages,
                    response_format=response_format,
                    temperature=0.7,
                )
            return completion.choices[0].message.parsed
        except Exception as e:
            bt.logging.warning(f"Error calling OpenAI: {e}")
            continue
    raise Exception("Failed to call OpenAI")
