import bittensor as bt
import numpy as np
import random
from typing import List, Tuple

from webgenie.helpers.htmls import html_to_screenshot, preprocess_html
from webgenie.protocol import WebgenieImageSynapse
from webgenie.tasks.solution import Solution
from webgenie.tasks.task import Task, ImageTask
from webgenie.tasks.task_generator import TaskGenerator
from webgenie.rewards.visual_reward import VisualReward
from webgenie.datasets.mockup_dataset import MockUpDataset
from webgenie.datasets.synthetic_dataset import SyntheticDataset

class ImageTaskGenerator(TaskGenerator):
    def __init__(self):
        super().__init__()
        self.rewards = [
            (VisualReward(), 1.0)
        ]
        self.datasets = [
            MockUpDataset(),
            SyntheticDataset()
        ]

    async def generate_task(self) -> Tuple[Task, bt.Synapse]:
        dataset_entry = await random.choice(self.datasets).generate_context()
        ground_truth_html = preprocess_html(dataset_entry.ground_truth_html)
        if ground_truth_html == "":
            raise ValueError("Invalid ground truth html")
        
        base64_image = html_to_screenshot(ground_truth_html)
        return ImageTask(
            base64_image=base64_image, 
            ground_truth_html=ground_truth_html,
            timeout=50,
            generator=self,
        ), WebgenieImageSynapse(base64_image=base64_image)

