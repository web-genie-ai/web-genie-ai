import bittensor as bt
import numpy as np
from typing import List, Tuple

from webgenie.rewards.incentive_rewards import get_incentive_rewards
from webgenie.tasks import Task
from webgenie.tasks.solution import Solution

class TaskGenerator:
    """
    A singleton generator for tasks.
    """
    def __init__(self):
        pass       

    async def generate_task(self) -> Tuple[Task, bt.Synapse]:
        pass
    
    async def reward(self, task: Task, solutions: List[Solution]) -> np.ndarray:
        scores = np.zeros(len(solutions))
        for reward, weight in self.rewards:
            reward_scores = await reward.reward(task, solutions)
            scores += weight * np.array(reward_scores)
        rewards = get_incentive_rewards(scores)
        return rewards

