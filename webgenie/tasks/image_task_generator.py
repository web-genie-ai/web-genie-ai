import bittensor as bt
import numpy as np
import hashlib
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
    CentralDataset,
)

class ImageTaskGenerator(TaskGenerator):    
    def __init__(self):
        super().__init__()
        
        self.datasets = [
            (CentralDataset(), 1),
            #(RandomWebsiteDataset(), 1),
            #(SyntheticDataset(), 0.5),
            #(HuggingfaceDataset(dataset_name="SALT-NLP/Design2Code-hf", split="train", html_column="text"), 1),
        ]

        self.metrics = {
            ACCURACY_METRIC_NAME: VisualReward(),
            SEO_METRIC_NAME: LighthouseReward(),
            QUALITY_METRIC_NAME: QualityReward(),
        }

    async def generate_task(self, **kwargs) -> Tuple[Task, bt.Synapse]:
        bt.logging.info("Generating Image task")
        
        dataset, _ = random.choices(self.datasets, weights=[weight for _, weight in self.datasets])[0]
        dataset_entry = await dataset.generate_context(**kwargs)
        bt.logging.debug(f"Generated dataset entry: {dataset_entry.url}")

        ground_truth_html = preprocess_html(dataset_entry.ground_truth_html)
        bt.logging.info(f"Preprocessed ground truth html")
        if not ground_truth_html :
            raise ValueError("Invalid ground truth html")

        if is_empty_html(ground_truth_html):
            raise ValueError("Empty ground truth html")
        
        base64_image = await html_to_screenshot(ground_truth_html, page_load_time=GROUND_TRUTH_HTML_LOAD_TIME)   
        # Check image dimensions ratio
        image = base64_to_image(base64_image)
        width, height = image.size
        aspect_ratio = height / width
        if aspect_ratio > 7:  # If height is more than 7x the width
            raise ValueError(f"Image aspect ratio too extreme: {aspect_ratio:.2f}. Height should not exceed 7x width.")
        
        bt.logging.debug(f"Screenshot generated for {dataset_entry.url}")
        image_task = ImageTask(
            base64_image=base64_image,
            ground_truth_html=ground_truth_html,
            generator=self,
            src=dataset_entry.src,
            task_id=hashlib.sha256(dataset_entry.url.encode()).hexdigest(),
            timeout=IMAGE_TASK_TIMEOUT,
        )
        
        return (
            image_task,  
            WebgenieImageSynapse(base64_image=base64_image, task_id=image_task.task_id),
        )
