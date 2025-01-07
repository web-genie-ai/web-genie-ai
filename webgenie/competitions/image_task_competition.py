import bittensor as bt
import numpy as np
import random
from typing import List, Tuple

from webgenie.constants import IMAGE_TASK_TIMEOUT
from webgenie.helpers.htmls import html_to_screenshot, preprocess_html, is_empty_html
from webgenie.protocol import WebgenieImageSynapse
from webgenie.tasks.solution import Solution
from webgenie.tasks.task import Task, ImageTask
from webgenie.competitions.competition import Competition
from webgenie.rewards.quality_reward import QualityReward
from webgenie.rewards.visual_reward import VisualReward
from webgenie.datasets import (
    RandomWebsiteDataset,
    SyntheticDataset,
    HuggingfaceDataset,
)

class ImageTaskCompetition(Competition):
    name = "ImageTaskCompetition"
    def __init__(self):
        super().__init__()
        
        self.datasets = [
            RandomWebsiteDataset(),
            #SyntheticDataset(),
            #HuggingfaceDataset(dataset_name="SALT-NLP/Design2Code-hf", split="train", html_column="text"),
        ]

    async def generate_task(self) -> Tuple[Task, bt.Synapse]:
        bt.logging.info("Generating Image task")
        dataset_entry = await random.choice(self.datasets).generate_context()
        ground_truth_html = preprocess_html(dataset_entry.ground_truth_html)
        if not ground_truth_html :
            raise ValueError("Invalid ground truth html")

        if is_empty_html(ground_truth_html):
            raise ValueError("Empty ground truth html")
        
        base64_image = html_to_screenshot(ground_truth_html)
        return ImageTask(
            base64_image=base64_image, 
            ground_truth_html=ground_truth_html,
            timeout=IMAGE_TASK_TIMEOUT,
            competition=self,
        ), WebgenieImageSynapse(base64_image=base64_image)

class ImageTaskAccuracyCompetition(ImageTaskCompetition):
    name = "ImageTaskAccuracyCompetition"
    def __init__(self):
        super().__init__()

        self.rewards = [
            (VisualReward(), 0.9),
            (QualityReward(), 0.1)
        ]

class ImageTaskQualityCompetition(ImageTaskCompetition):
    name = "ImageTaskQualityCompetition"
    def __init__(self):
        super().__init__()

        self.rewards = [
            (VisualReward(), 0.5),
            (QualityReward(), 0.5)
        ]
