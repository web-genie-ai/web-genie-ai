# The paper [Unsupervised Evaluation of Code LLMs with Round-Trip Correctness]
# (https://arxiv.org/pdf/2402.08699#page=11&zoom=100,384,458) is our inspiration for this reward.

import bittensor as bt
import asyncio
import numpy as np
from pydantic import BaseModel, Field
from typing import List

from webgenie.helpers.llms import openai_call
from webgenie.prompts import PROMPT_QUALITY
from webgenie.rewards.reward import Reward
from webgenie.tasks import Task, Solution


class ScoreResponse(BaseModel):
    score: float = Field(description="The score of the html code")


class QualityReward(Reward):

    async def _get_score(self, solution: Solution) -> float:
        response = await openai_call(
            messages = [
                {"role": "system", "content": PROMPT_QUALITY.format(html=solution.html)},
            ],
            response_format = ScoreResponse,
            deterministic=True,
        )
        return response.score / 100

    async def reward(self, task: Task, solutions: List[Solution]) -> np.ndarray:
        bt.logging.info(f"Rewarding task in quality reward")
        get_score_tasks = [self._get_score(solution) for solution in solutions]
        scores = await asyncio.gather(*get_score_tasks)
        return np.array(scores)
