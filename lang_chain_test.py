from functools import partial
from starlette.types import Send
import time
from typing import Dict
import openai
import os

from typing import Dict, Awaitable
from langchain_openai import ChatOpenAI,OpenAI

from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSequence

import bittensor as bt

class DummySynapse:
    query: str
    timeout: float
class Self:
    pass

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
    
def miner_init(self):
    bt.logging.debug(f"Dummy Miner initialized")
    # Set openai key and other args
    self.model = ChatOpenAI(
        api_key=api_key,
        model_name="gpt-4",
    )

def miner_forward(self, synapse: DummySynapse) -> str:
    def _forward(self, chain: RunnableSequence, chain_formatter: Dict[str, str], timeout_threshold: float, init_time: float) -> str:
        buffer = []
        timeout_reached = False
        is_in_code_block = True
        is_first_line = True
        generated_code = ""
        try:
            for token in chain.stream(chain_formatter):
                
                if is_first_line and token.startswith("```"):
                    is_in_code_block = False

                if is_in_code_block and not token.startswith("```"):
                    pass
                buffer.append(token)

                if time.time() - init_time > timeout_threshold:
                    bt.logging.debug(f"⏰ Timeout reached, stopping streaming")
                    timeout_reached = True
                    break

                if len(buffer) == 10:
                    joined_buffer = "".join(buffer)
                    generated_code += joined_buffer
                    buffer = []

                if is_first_line and token.endswith("\n"):
                    is_first_line = False
                    is_in_code_block = True

            if (
                buffer and not timeout_reached
            ):  # Don't send the last buffer of data if timeout.
                joined_buffer = "".join(buffer)
                generated_code += joined_buffer
            return generated_code
        except Exception as e:
            bt.logging.error(f"Dummy Miner Error: {e}")
            return ""
    bt.logging.debug(f"Dummy Miner Query received, forwarding synapse: {synapse}")
    
    # prompt = PromptTemplate.from_template(
    #     "You are an expert programmer. Generate code based on the following request:\n\n{query}\n\nProvide only the code, without any explanations."
    # )
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert programmer. Generate code based on the following request without explanations:\n\n{query}\n\n Provide the code that satisfies the following requirements without any explanation. You must give me both the CSS file and the frontend HTML file for a 'Login Page.' The response should be in JSON format as shown below, and it should be plaintext (without triple quotes):
        {{ 'CSS': 'css_code_here', 'HTML': 'html_code_here' }}"""),
        ("user", "{query}"),
    ])
    chain = prompt | self.model | StrOutputParser()

    query = synapse.query
    bt.logging.info(f"Dummy Miner Query: {query}")

    chain_formatter = {"query": query}

    init_time = time.time()
    timeout_threshold = synapse.timeout
    token_streamer = partial(
        _forward,
        self,
        chain,
        chain_formatter,
        timeout_threshold,
        init_time,
    )
    return token_streamer()

def evaluate_code(task: str, generated_code: str) -> float:    
    # Use GPT to evaluate the code
    query = f"""
    Task: {task}
    Generated Code:
    {generated_code}

    Evaluate the generated code based on the following criteria:
    1. Correctness: Does the code implement the required functionality?
    2. Code quality: Is the code well-structured and following best practices?
    3. Completeness: Does the code address all aspects of the task?

    Provide a score between 0 and 1, where 1 is perfect and 0 is completely incorrect.
    Only return the score, without any explanations.
    """

    try:
        evaluation = ChatOpenAI(
            api_key=api_key,
            model="gpt-4",
            
        )
        messages=[
                {"role": "system", "content": "You are a code evaluation expert."},
                {"role": "user", "content": query}
            ]
        evaluation = evaluation.invoke(messages)
        return float(evaluation.content)
    except Exception as e:
        bt.logging.error(f"Error evaluating code: {e}")
        return 0.0

if __name__ == "__main__":
    self = Self()
    synapse = DummySynapse()
    synapse.query = "Comming soon Page using Vue.js and css"
    synapse.timeout = 50
    miner_init(self)
    generated_code = miner_forward(self, synapse)
    print(f"Generated Code: {generated_code}")
    score = evaluate_code(synapse.query, generated_code)
    print(f"Score: {score}")
