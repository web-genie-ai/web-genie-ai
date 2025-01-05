import os
import bittensor as bt

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

LLM = ChatOpenAI(
    base_url=os.getenv("LLM_MODEL_URL"),
    model=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    temperature=0.7
)

async def call_llm(template, params, output_parser, retries=3):
    if not os.getenv("LLM_API_KEY"):
        raise Exception("LLM_API_KEY is not set")
    
    for _ in range(retries):
        try:
            prompt = ChatPromptTemplate.from_messages(template)
            chain = prompt | LLM | output_parser
            return await chain.ainvoke(params)
        except Exception as e:
            bt.logging.error(f"Error calling LLM: {e}")
            continue
    raise Exception("Failed to call LLM")