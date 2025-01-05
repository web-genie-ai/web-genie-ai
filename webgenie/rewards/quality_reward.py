# The paper [Unsupervised Evaluation of Code LLMs with Round-Trip Correctness]
# (https://arxiv.org/pdf/2402.08699#page=11&zoom=100,384,458) is our inspiration for this reward.

import bittensor as bt
import bert_score
import os
import numpy as np
from typing import List

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from webgenie.prompts import PROMPT_QUALITY
from webgenie.rewards import Reward
from webgenie.rewards.metrics import s_bert
from webgenie.tasks.task import Task
from webgenie.tasks.solution import Solution


class ScoreResponse(BaseModel):
    score: float = Field(default=0, description="The score of the html code")

class QualityReward(Reward):
    def __init__(self):
        self.model = ChatOpenAI(
            api_key= os.getenv("LLM_API_KEY"),
            model_name=os.getenv("LLM_MODEL_ID"),
            base_url=os.getenv("LLM_MODEL_URL"),
        )

        self.prompt_response_parser = JsonOutputParser(pydantic_object=ScoreResponse)
        
    async def _get_score(self, solution: Solution) -> float:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(PROMPT_QUALITY)
        ])

        chain = prompt | self.model | self.prompt_response_parser
        prompt_response = await chain.ainvoke({
            "html": solution.html,
            "instructions": self.prompt_response_parser.get_format_instructions()
        })

        return float(prompt_response["score"]) / 100

    async def reward(self, task: Task, solutions: List[Solution]) -> np.ndarray:
        bt.logging.debug(f"Rewarding task in quality reward")
        scores = []
        for solution in solutions:
            score = await self._get_score(solution)
            scores.append(score)
        return np.array(scores)



        