import bittensor as bt
import numpy as np
import random
from typing import Tuple, List

from webgenie.competitions.competition import (
    Competition,
    ACCURACY_METRIC_NAME,
    QUALITY_METRIC_NAME,
)
from webgenie.constants import TEXT_TASK_TIMEOUT
from webgenie.datasets import (
    SyntheticDataset,
)
from webgenie.protocol import WebgenieTextSynapse
from webgenie.rewards import (
    QualityReward,
    RtcReward,
)
from webgenie.tasks import Task, TextTask, Solution


class TextTaskCompetition(Competition):
    COMPETITION_TYPE = "TextTaskCompetition"

    def __init__(self):
        super().__init__()
    
        self.datasets = [
            (SyntheticDataset(has_ground_truth_html = False), 0.8),
        ]

        self.metrics = {
            ACCURACY_METRIC_NAME: RtcReward(),
            QUALITY_METRIC_NAME: QualityReward(),
        }

    async def generate_task(self) -> Tuple[Task, bt.Synapse]:
        bt.logging.info("Generating Text task")
        dataset, _ = random.choices(self.datasets, weights=[weight for _, weight in self.datasets])[0]
        dataset_entry = await dataset.generate_context()
        
        return (
            TextTask(
                prompt=dataset_entry.prompt, 
                ground_truth_html=dataset_entry.ground_truth_html,
                timeout=TEXT_TASK_TIMEOUT,
                competition=self,
            ), 
            WebgenieTextSynapse(prompt=dataset_entry.prompt),
        )


class TextTaskAccuracyCompetition(TextTaskCompetition):
    COMPETITION_TYPE = "TextTaskAccuracyCompetition"

    async def calculate_final_scores(self, task: Task, solutions: List[Solution]) -> np.ndarray:
        scores = await self.calculate_scores(task, solutions)
        return scores[ACCURACY_METRIC_NAME] * 0.9 + scores[QUALITY_METRIC_NAME] * 0.1, scores


class TextTaskQualityCompetition(TextTaskCompetition):
    COMPETITION_TYPE = "TextTaskQualityCompetition"

    async def calculate_final_scores(self, task: Task, solutions: List[Solution]) -> np.ndarray:
        scores = await self.calculate_scores(task, solutions)
        accuracy_scores = scores[ACCURACY_METRIC_NAME]
        quality_scores = scores[QUALITY_METRIC_NAME]
        final_scores = np.where(accuracy_scores > 0.7, quality_scores, 0)
        return final_scores, scores
