import bittensor as bt
import numpy as np
from typing import List, Tuple

from webgenie.rewards import Reward
from webgenie.tasks.solution import Solution
from webgenie.tasks.task import Task


class TaskGenerator:
    def __init__(self):
        self.metrics: dict[str, Reward] = {}

    async def generate_task(self) -> Tuple[Task, bt.Synapse]:
        pass
    
    async def calculate_scores(self, task: Task, solutions: List[Solution]) -> dict[str, np.ndarray]:
        scores: dict[str, np.ndarray] = {}
        for metric_name, reward_model in self.metrics.items():
            reward_scores = await reward_model.reward(task, solutions)
            scores[metric_name] = reward_scores
        return scores
