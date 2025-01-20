import bittensor as bt
import numpy as np
import random
from typing import Tuple, List

from webgenie.tasks.metric_types import (
    ACCURACY_METRIC_NAME,
    QUALITY_METRIC_NAME,
)
from webgenie.tasks.task_generator import TaskGenerator
from webgenie.constants import TEXT_TASK_TIMEOUT
from webgenie.datasets import (
    SyntheticDataset,
)
from webgenie.protocol import WebgenieTextSynapse
from webgenie.rewards import (
    QualityReward,
    RtcReward,
)
from webgenie.tasks.solution import Solution
from webgenie.tasks.task import Task, TextTask


class TextTaskGenerator(TaskGenerator):
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
                generator=self,
            ), 
            WebgenieTextSynapse(prompt=dataset_entry.prompt),
        )
