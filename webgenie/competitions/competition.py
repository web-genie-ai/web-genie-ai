import bittensor as bt
import numpy as np
from typing import List, Tuple

from webgenie.rewards import Reward
from webgenie.tasks import Task, Solution


ACCURACY_METRIC_NAME = "Accuracy"
SEO_METRIC_NAME = "Seo"
QUALITY_METRIC_NAME = "Quality"


class Competition:
    COMPETITION_TYPE = "UnknownCompetition"
    
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

    async def calculate_final_scores(self, task: Task, solutions: List[Solution]) -> Tuple[np.ndarray, dict[str, np.ndarray]]:
        pass

