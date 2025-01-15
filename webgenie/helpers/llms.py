import os
import bittensor as bt

from openai import AsyncOpenAI

model = os.getenv("LLM_MODEL_ID")
api_key = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_MODEL_URL")

if not api_key or not base_url or not model:
    raise Exception("LLM_API_KEY, LLM_MODEL_URL, and LLM_MODEL_ID must be set")

client = AsyncOpenAI(
    api_key=api_key,
    base_url=base_url,
)

async def openai_call(messages, response_format, deterministic=False, retries=3):
    for _ in range(retries):
        try:
            if deterministic:
                completion = await client.beta.chat.completions.parse(
                    model=model,
                    messages= messages,
                    response_format=response_format,
                    temperature=0,
                )
            else:
                completion = await client.beta.chat.completions.parse(
                    model=model,
                    messages= messages,
                    response_format=response_format,
                    temperature=0.7,
                )
            return completion.choices[0].message.parsed
        except Exception as e:
            bt.logging.error(f"Error calling OpenAI: {e}")
            continue
    raise Exception("Failed to call OpenAI")