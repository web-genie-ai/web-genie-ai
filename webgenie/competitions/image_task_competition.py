import bittensor as bt
import random
from typing import Tuple

from webgenie.competitions.competition import Competition
from webgenie.constants import IMAGE_TASK_TIMEOUT
from webgenie.helpers.htmls import (
    html_to_screenshot, 
    preprocess_html, 
    is_empty_html,
)
from webgenie.helpers.images import base64_to_image
from webgenie.protocol import WebgenieImageSynapse
from webgenie.tasks import Task, ImageTask
from webgenie.rewards import (
    QualityReward,
    VisualReward,
)
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
        
        # Save base64_image for debugging purposes
        import os
        import base64
        from datetime import datetime

        debug_dir = "debug_images"
        os.makedirs(debug_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(debug_dir, f"image_{timestamp}.png")
        
        try:
            image = base64_to_image(base64_image)
            image.save(filename)
        except Exception as e:
            bt.logging.error(f"Failed to save debug image: {e}")
        
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
            (QualityReward(), 0.1),
        ]


class ImageTaskQualityCompetition(ImageTaskCompetition):
    name = "ImageTaskQualityCompetition"
    def __init__(self):
        super().__init__()

        self.rewards = [
            (VisualReward(), 0.5),
            (QualityReward(), 0.5),
        ]
