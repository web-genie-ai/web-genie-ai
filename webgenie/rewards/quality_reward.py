# The paper [Unsupervised Evaluation of Code LLMs with Round-Trip Correctness]
# (https://arxiv.org/pdf/2402.08699#page=11&zoom=100,384,458) is our inspiration for this reward.

import bittensor as bt
import numpy as np
from typing import List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from webgenie.helpers.llms import call_llm
from webgenie.prompts import PROMPT_QUALITY
from webgenie.rewards.reward import Reward
from webgenie.tasks import Task, Solution


class ScoreResponse(BaseModel):
    score: float = Field(default=0, description="The score of the html code")


class QualityReward(Reward):
    def __init__(self):
        self.prompt_response_parser = JsonOutputParser(pydantic_object=ScoreResponse)
        
    async def _get_score(self, solution: Solution) -> float:
        response = await call_llm(
            template=[
                ("system", PROMPT_QUALITY),
            ],
            params={
                "html": solution.html, 
                "instructions": self.prompt_response_parser.get_format_instructions(),
            },
            output_parser=self.prompt_response_parser,
        )
        return float(response["score"]) / 100

    async def reward(self, task: Task, solutions: List[Solution]) -> np.ndarray:
        bt.logging.debug(f"Rewarding task in quality reward")
        scores = []
        for solution in solutions:
            score = await self._get_score(solution)
            scores.append(score)
        return np.array(scores)



        