import bittensor as bt
import numpy as np
from typing import List, Tuple

from webgenie.tasks import Task, Solution

class Competition:
    name = "Competition"
    def __init__(self):
        self.rewards = []

    async def generate_task(self) -> Tuple[Task, bt.Synapse]:
        pass
    
    async def reward(self, task: Task, solutions: List[Solution]) -> np.ndarray:
        scores = np.zeros(len(solutions))
        for reward, weight in self.rewards:
            reward_scores = await reward.reward(task, solutions)
            scores += weight * np.array(reward_scores)
        return scores

