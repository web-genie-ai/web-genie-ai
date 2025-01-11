# The paper [Unsupervised Evaluation of Code LLMs with Round-Trip Correctness]
# (https://arxiv.org/pdf/2402.08699#page=11&zoom=100,384,458) is our inspiration for this reward.

import bittensor as bt
import numpy as np
from typing import List

from webgenie.rewards.reward import Reward
from webgenie.tasks import Task, Solution

from .get_lighthouse_score import get_lighthouse_score


class LighthouseReward(Reward):

    async def reward(self, task: Task, solutions: List[Solution]) -> np.ndarray:
        bt.logging.info(f"Rewarding lighthouse task")
        htmls = [solution.html for solution in solutions]
        scores_dict = get_lighthouse_score(htmls)
        scores = []
        weights = [0, 0.25, 0.25, 0.5]
        for score_dict in scores_dict:
            score = (
                score_dict['performance'] * weights[0] + 
                score_dict['accessibility'] * weights[1] + 
                score_dict['best-practices'] * weights[2] + 
                score_dict['seo'] * weights[3]
            )
            scores.append(score)
        return np.array(scores)
