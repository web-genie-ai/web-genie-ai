import bittensor as bt
import numpy as np
import random
from typing import Tuple, List

from webgenie.tasks.metric_types import (
    ACCURACY_METRIC_NAME, 
    QUALITY_METRIC_NAME,
    SEO_METRIC_NAME,
)
from webgenie.tasks.task_generator import TaskGenerator
from webgenie.constants import IMAGE_TASK_TIMEOUT, GROUND_TRUTH_HTML_LOAD_TIME
from webgenie.helpers.htmls import (
    html_to_screenshot, 
    preprocess_html, 
    is_empty_html,
)
from webgenie.helpers.images import base64_to_image
from webgenie.protocol import WebgenieImageSynapse
from webgenie.tasks.solution import Solution
from webgenie.tasks.task import Task, ImageTask
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


class ImageTaskGenerator(TaskGenerator):    
    def __init__(self):
        super().__init__()
        
        self.datasets = [
            (RandomWebsiteDataset(), 1),
            #(SyntheticDataset(), 0.1),
            #(HuggingfaceDataset(dataset_name="SALT-NLP/Design2Code-hf", split="train", html_column="text"), 0.1),
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
        bt.logging.debug(f"Generated dataset entry: {dataset_entry.src}")

        ground_truth_html = preprocess_html(dataset_entry.ground_truth_html)
        bt.logging.info(f"Preprocessed ground truth html")
        if not ground_truth_html :
            raise ValueError("Invalid ground truth html")

        if is_empty_html(ground_truth_html):
            raise ValueError("Empty ground truth html")
        
        base64_image = await html_to_screenshot(ground_truth_html, page_load_time=GROUND_TRUTH_HTML_LOAD_TIME)    
        bt.logging.debug(f"Screenshot generated for {dataset_entry.src}")
        image_task = ImageTask(
            base64_image=base64_image, 
            ground_truth_html=ground_truth_html,
            timeout=IMAGE_TASK_TIMEOUT,
            generator=self,
            src=dataset_entry.src,
        )
        return (
            image_task,  
            WebgenieImageSynapse(base64_image=base64_image, task_id=image_task.task_id),
        )
