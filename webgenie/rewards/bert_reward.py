import bert_score
import bittensor as bt
import numpy as np
from typing import List

from webgenie.rewards.reward import Reward
from webgenie.tasks import (
    Task,
    Solution,
)


class BertReward(Reward):
    def __init__(self):
        pass

    async def reward(self, task: Task, solutions: List[Solution]) -> np.ndarray:
        bt.logging.info(f"Rewarding task in bert reward")
        if not task.ground_truth_html:
            raise ValueError(f"Ground truth html is empty")
        
        original_htmls= []
        miner_htmls = []

        for solution in solutions:
            original_htmls.append(task.ground_truth_html)
            miner_htmls.append(solution.html)

        P, R, F1 = bert_score.score(original_htmls, miner_htmls, lang='en')

        return np.array(F1)

        