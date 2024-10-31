import json
from functools import partial
from starlette.types import Send
import time
from typing import Dict

from typing import Dict, Awaitable
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate    
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSequence

import bittensor as bt
import btcopilot
import os

def miner_init(self):
    bt.logging.debug(f"Dummy Miner initialized")
    _ = load_dotenv(find_dotenv())
    api_key = os.getenv("OPENAI_API_KEY")
    # Set openai key and other args
    self.model = ChatOpenAI(
        api_key=api_key,
        model_name="gpt-4",
    )

def miner_forward(self, synapse: btcopilot.protocol.BtCopilotSynapse)->Awaitable:
    
    async def _forward(self, chain: RunnableSequence, chain_formatter: Dict[str, str], timeout_threshold: float, init_time: float, send: Send):
        try:
            json_response = {
                "css": "body { background-color: red; }",
                "html": "<html><body><h1>Hello World</h1></body></html>",
            }
            await send(
                {
                    "type": "http.response.body",
                    "body": json.dumps(json_response).encode("utf-8"),
                    "more_body": False,
                }
            )
            # buffer = []

            # timeout_reached = False
            # is_in_code_block = True
            # is_first_line = True
            
            # for token in chain.stream(chain_formatter):
            #     if is_first_line and token.startswith("```"):
            #         is_in_code_block = False

            #     if is_in_code_block and not token.startswith("```"):
            #         buffer.append(token)

            #     if time.time() - init_time > timeout_threshold:
            #         bt.logging.debug(f"⏰ Timeout reached, stopping streaming")
            #         timeout_reached = True
            #         break

            #     if len(buffer) == self.config.neuron.streaming_batch_size:
            #         joined_buffer = "".join(buffer)
            #         bt.logging.debug(f"Streamed tokens: {joined_buffer}")

            #         await send(
            #             {
            #                 "type": "http.response.body",
            #                 "body": joined_buffer.encode("utf-8"),
            #                 "more_body": True,
            #             }
            #         )
            #         buffer = []
                
            #     if is_first_line and token.endswith("\n"):
            #         is_first_line = False
            #         is_in_code_block = True

            # if (
            #     buffer and not timeout_reached
            # ):  # Don't send the last buffer of data if timeout.
            #     joined_buffer = "".join(buffer)
            #     await send(
            #         {
            #             "type": "http.response.body",
            #             "body": joined_buffer.encode("utf-8"),
            #             "more_body": False,
            #         }
            #     )
        except Exception as e:
            bt.logging.error(f"Dummy Miner Error: {e}")

    bt.logging.debug(f"Dummy Miner Query received, forwarding synapse: {synapse}")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert programmer. Generate code based on the following request without explanations:\n\n{query}\n\nProvide only the code."),
        ("user", "{query}"),
    ])
    chain = prompt | self.model | StrOutputParser()

    query = synapse.task.query
    bt.logging.debug(f"Dummy Miner Query received: {synapse}")
    time.sleep(2)
    chain_formatter = {"query": query}

    init_time = time.time()
    timeout_threshold = float(synapse.timeout)

    token_streamer = partial(
        _forward,
        self,
        chain,
        chain_formatter,
        timeout_threshold,
        init_time,
    )
    return synapse.create_streaming_response(token_streamer)  