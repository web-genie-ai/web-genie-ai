import bittensor as bt
import bert_score
import os
import numpy as np
from typing import List

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from webgenie.prompts import PROMPT_RTC
from webgenie.rewards import Reward
from webgenie.tasks.task import Task
from webgenie.tasks.solution import Solution

class PromptResponse(BaseModel):
    prompt: str = Field(default="", description="The prompt that generates the given html code")

class RtcReward(Reward):
    def __init__(self):
        self.model = ChatOpenAI(
            api_key= os.getenv("OPENAI_API_KEY"),
            model_name="gpt-4o",
        )

        self.prompt_response_parser = JsonOutputParser(pydantic_object=PromptResponse)

    def _get_prompt(self, task: Task, solutions: List[Solution]) -> str:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(PROMPT_RTC)
        ])

        chain = prompt | self.model | self.prompt_response_parser
        prompt_response = chain.invoke({
            "html": task.ground_truth_html,
            "prompt": task.prompt,
            "instructions": self.prompt_response_parser.get_format_instructions()
        })

        return prompt_response["prompt"]

    async def reward(self, task: Task, solutions: List[Solution]) -> np.ndarray:
        bt.logging.debug(f"Rewarding task in rtc reward")
        original_prompts = [task.prompt for _ in solutions]
        miner_prompts = [self._get_prompt(task, solution) for solution in solutions]
        P, R, F1 = bert_score.score(original_prompts, miner_prompts, lang='en')
        return np.array(R)



        