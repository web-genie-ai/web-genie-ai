import bittensor as bt
import numpy as np
import random
from typing import List, Tuple

from webgenie.helpers.htmls import html_to_screenshot, preprocess_html, is_empty_html
from webgenie.protocol import WebgenieImageSynapse
from webgenie.tasks.solution import Solution
from webgenie.tasks.task import Task, ImageTask
from webgenie.tasks.task_generator import TaskGenerator
from webgenie.rewards.visual_reward import VisualReward
from webgenie.datasets.mockup_dataset import MockUpDataset
from webgenie.datasets.synthetic_dataset import SyntheticDataset
from webgenie.datasets.huggingface_dataset import HuggingfaceDataset    

class ImageTaskGenerator(TaskGenerator):
    def __init__(self):
        super().__init__()
        self.rewards = [
            (VisualReward(), 1.0)
        ]
        self.datasets = [
        #    MockUpDataset(),
            SyntheticDataset(),
            HuggingfaceDataset("SALT-NLP/Design2Code-hf", "train", "text"),
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
            timeout=250,
            generator=self,
        ), WebgenieImageSynapse(base64_image=base64_image)

