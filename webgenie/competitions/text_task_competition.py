import bittensor as bt
import numpy as np
import random
from typing import List, Tuple
from webgenie.datasets import (
    SyntheticDataset,
)
from webgenie.constants import TEXT_TASK_TIMEOUT
from webgenie.protocol import WebgenieTextSynapse
from webgenie.rewards.quality_reward import QualityReward
from webgenie.rewards.rtc_reward import RtcReward
from webgenie.tasks.task import Task, TextTask
from webgenie.competitions.competition import Competition

class TextTaskCompetition(Competition):
    name = "TextTaskCompetition"
    def __init__(self):
        super().__init__()
    
        self.datasets = [
            SyntheticDataset(has_ground_truth_html = True)
        ]
    
    async def generate_task(self) -> Tuple[Task, bt.Synapse]:
        bt.logging.info("Generating Text task")
        dataset_entry = await random.choice(self.datasets).generate_context()
        return TextTask(
            prompt=dataset_entry.prompt, 
            ground_truth_html=dataset_entry.ground_truth_html,
            timeout=TEXT_TASK_TIMEOUT,
            competition=self
        ), WebgenieTextSynapse(prompt=dataset_entry.prompt)

class TextTaskAccuracyCompetition(TextTaskCompetition):
    name = "TextTaskAccuracyCompetition"
    def __init__(self):
        super().__init__()
        self.rewards = [
            (RtcReward(), 0.9),
            (QualityReward(), 0.1)
        ]

class TextTaskQualityCompetition(TextTaskCompetition):
    name = "TextTaskQualityCompetition"
    def __init__(self):
        super().__init__()
        self.rewards = [
            (RtcReward(), 0.5),
            (QualityReward(), 0.5)
        ]
