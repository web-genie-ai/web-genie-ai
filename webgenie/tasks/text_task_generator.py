import bittensor as bt
import numpy as np
import random
from typing import List, Tuple

from webgenie.datasets.dataset import MockUpPromptDataset
from webgenie.protocol import WebgenieTextSynapse
from webgenie.tasks.solution import Solution
from webgenie.tasks.task import Task, TextTask
from webgenie.tasks.task_generator import TaskGenerator

class TextTaskGenerator(TaskGenerator):
    def __init__(self):
        super().__init__()
        self.rewards = []
        self.datasets = [
            MockUpPromptDataset()
        ]

    async def generate_task(self) -> Tuple[Task, bt.Synapse]:
        dataset_entry = await random.choice(self.datasets).generate_context()
        return TextTask(
            prompt=dataset_entry.prompt, 
            timeout=50,
            generator=self
        ), WebgenieTextSynapse(prompt=dataset_entry.prompt)
