import bert_score
import bittensor as bt
import numpy as np
from typing import List

from webgenie.rewards import Reward
from webgenie.tasks.task import Task
from webgenie.tasks.solution import Solution

class BertReward(Reward):
    def __init__(self):
        pass

    async def reward(self, task: Task, solutions: List[Solution]) -> np.ndarray:
        bt.logging.debug(f"Rewarding task in bert reward")

        original_htmls= []
        miner_htmls = []

        for solution in solutions:
            original_htmls.append(task.ground_truth_html)
            miner_htmls.append(solution.html)

        P, R, F1 = bert_score.score(original_htmls, miner_htmls, lang='en')

        return np.array(F1)

        