from init_test import init_test
init_test()

import asyncio
import numpy as np
from typing import List
import time
from webgenie.tasks import Task, Solution, ImageTask
from webgenie.rewards import (
    LighthouseReward,
    QualityReward,
    VisualReward,
)

from webgenie.protocol import WebgenieImageSynapse
from webgenie.competitions.competition import (
    ACCURACY_METRIC_NAME,
    SEO_METRIC_NAME,
    QUALITY_METRIC_NAME,
)

from webgenie.helpers.htmls import html_to_screenshot
from neurons.miners.openai_miner import OpenaiMiner

metrics = {
    ACCURACY_METRIC_NAME: VisualReward(),
    SEO_METRIC_NAME: LighthouseReward(),
    QUALITY_METRIC_NAME: QualityReward(),
}

async def calculate_scores(task: Task, solutions: List[Solution]) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}

    for metric_name, reward_model in metrics.items():
        print(metric_name)
        start_time = time.time()
        reward_scores = await reward_model.reward(task, solutions)
        execution_time = time.time() - start_time
        print(f"Execution time: {execution_time:.2f} seconds")
        scores[metric_name] = reward_scores
        print(scores[metric_name])
    return scores


async def main():
    ground_truth_html_path = "tests/work/original.html"
    with open(ground_truth_html_path, "r") as f:
        ground_truth_html = f.read()
    
    print("HTML to screenshot")
    start_time = time.time()
    base64_image = await html_to_screenshot(ground_truth_html)
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    
    task = ImageTask(
        ground_truth_html=ground_truth_html,
    )

    miner = OpenaiMiner(neuron=None)
    synapse = WebgenieImageSynapse(
        base64_image = base64_image,
    )

    print("Miner forward image")
    start_time = time.time()
    synapse = await miner.forward_image(synapse)
    print(synapse.html)
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    solutions = [Solution(html=synapse.html) for _ in range(1)]

    print("Calculate scores")
    start_time = time.time()
    scores = await calculate_scores(task, solutions)
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    
    print(scores)

if __name__ == "__main__":
    asyncio.run(main())
