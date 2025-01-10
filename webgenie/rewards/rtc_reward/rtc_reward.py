# The paper [Unsupervised Evaluation of Code LLMs with Round-Trip Correctness]
# (https://arxiv.org/pdf/2402.08699#page=11&zoom=100,384,458) is our inspiration for this reward.

import bittensor as bt
import bert_score
import os
import numpy as np
from pydantic import BaseModel, Field
from typing import List


from webgenie.helpers.llms import openai_call
from webgenie.prompts import PROMPT_RTC
from webgenie.rewards.reward import Reward
from webgenie.rewards.metrics import s_bert
from webgenie.tasks import Task, Solution


class PromptResponse(BaseModel):
    prompt: str = Field(default="", description="The prompt that generates the given html code")


class RtcReward(Reward):

    async def _get_prompt(self, task: Task, solution: Solution) -> str:
        response = await openai_call(
            messages = [
                {"role": "system", "content": PROMPT_RTC.format(html=solution.html, prompt=task.prompt)},
            ],
            response_format = PromptResponse,
        )

        return response["prompt"]

    async def reward(self, task: Task, solutions: List[Solution]) -> np.ndarray:
        bt.logging.debug(f"Rewarding task in rtc reward")
        original_prompts = [task.prompt for _ in solutions]
        miner_prompts = [await self._get_prompt(task, solution) for solution in solutions]
        
        #P, R, F1 = bert_score.score(original_prompts, miner_prompts, lang='en')
        scores = s_bert.score(original_prompts, miner_prompts)
        return np.array(scores)
