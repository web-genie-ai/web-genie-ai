import bittensor as bt
import numpy as np
import random
from typing import Tuple, List

from webgenie.competitions.competition import (
    Competition, 
    ACCURACY_METRIC_NAME, 
    QUALITY_METRIC_NAME,
    SEO_METRIC_NAME,
)
from webgenie.constants import IMAGE_TASK_TIMEOUT, GROUND_TRUTH_HTML_LOAD_TIME
from webgenie.helpers.htmls import (
    html_to_screenshot, 
    preprocess_html, 
    is_empty_html,
)
from webgenie.helpers.images import base64_to_image
from webgenie.protocol import WebgenieImageSynapse
from webgenie.tasks import Task, ImageTask, Solution
from webgenie.rewards import (
    QualityReward,
    VisualReward,
    LighthouseReward,
)
from webgenie.datasets import (
    RandomWebsiteDataset,
    SyntheticDataset,
    HuggingfaceDataset,
)


class ImageTaskCompetition(Competition):
    COMPETITION_TYPE = "ImageTaskCompetition"
    
    def __init__(self):
        super().__init__()
        
        self.datasets = [
            (RandomWebsiteDataset(), 0.8),
            (SyntheticDataset(), 0.1),
            (HuggingfaceDataset(dataset_name="SALT-NLP/Design2Code-hf", split="train", html_column="text"), 0.1),
        ]

        self.metrics = {
            ACCURACY_METRIC_NAME: VisualReward(),
            SEO_METRIC_NAME: LighthouseReward(),
            QUALITY_METRIC_NAME: QualityReward(),
        }

    async def generate_task(self) -> Tuple[Task, bt.Synapse]:
        bt.logging.info("Generating Image task")
        
        dataset, _ = random.choices(self.datasets, weights=[weight for _, weight in self.datasets])[0]
        dataset_entry = await dataset.generate_context()

        ground_truth_html = preprocess_html(dataset_entry.ground_truth_html)
        if not ground_truth_html :
            raise ValueError("Invalid ground truth html")

        if is_empty_html(ground_truth_html):
            raise ValueError("Empty ground truth html")
        
        base64_image = html_to_screenshot(ground_truth_html, page_load_time=GROUND_TRUTH_HTML_LOAD_TIME)
        
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
        
        return (
            ImageTask(
                base64_image=base64_image, 
                ground_truth_html=ground_truth_html,
                timeout=IMAGE_TASK_TIMEOUT,
                competition=self,
            ), 
            WebgenieImageSynapse(base64_image=base64_image),
        )


class ImageTaskAccuracyCompetition(ImageTaskCompetition):
    COMPETITION_TYPE = "ImageTaskAccuracyCompetition"

    async def calculate_final_scores(self, task: Task, solutions: List[Solution]) -> np.ndarray:
        scores = await self.calculate_scores(task, solutions)
        return scores[ACCURACY_METRIC_NAME] * 0.9 + scores[QUALITY_METRIC_NAME] * 0.1, scores


class ImageTaskQualityCompetition(ImageTaskCompetition):
    COMPETITION_TYPE = "ImageTaskQualityCompetition"
    
    async def calculate_final_scores(self, task: Task, solutions: List[Solution]) -> np.ndarray:
        scores = await self.calculate_scores(task, solutions)        
        accuracy_scores = scores[ACCURACY_METRIC_NAME]
        quality_scores = scores[QUALITY_METRIC_NAME]
        final_scores = np.where(accuracy_scores > 0.7, quality_scores, 0)
        return final_scores, scores

class ImageTaskSeoCompetition(ImageTaskCompetition):
    COMPETITION_TYPE = "ImageTaskSeoCompetition"

    async def calculate_final_scores(self, task: Task, solutions: List[Solution]) -> np.ndarray:
        scores = await self.calculate_scores(task, solutions)
        accuracy_scores = scores[ACCURACY_METRIC_NAME]
        seo_scores = scores[SEO_METRIC_NAME]
        final_scores = np.where(accuracy_scores > 0.7, seo_scores, 0)
        return final_scores, scores
