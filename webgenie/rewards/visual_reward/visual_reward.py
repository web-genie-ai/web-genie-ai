# The paper [Design2Code: Benchmarking Multimodal Code Generation for Automated Front-End Engineering]
# (https://arxiv.org/pdf/2403.03163) is our inspiration for this reward.

import bittensor as bt
import numpy as np
from typing import List
import uuid

from webgenie.constants import WORK_DIR
from webgenie.rewards.reward import Reward
from webgenie.rewards.visual_reward.common.browser import start_browser, stop_browser
from webgenie.rewards.visual_reward.high_level_matching_score import high_level_matching_score
from webgenie.rewards.visual_reward.low_level_matching_score import low_level_matching_score
from webgenie.tasks import Task, ImageTask, Solution


class VisualReward(Reward):
    def __init__(self):
        pass

    async def reward(self, task: Task, solutions: List[Solution]) -> np.ndarray:
        if not isinstance(task, ImageTask):
            raise ValueError(f"Task is not a ImageTask: {type(task)}")
        await start_browser()
        bt.logging.info(f"Rewarding image task in visual reward")
        
        original_html_path = f"{WORK_DIR}/original_{uuid.uuid4()}.html"
        with open(original_html_path, "w") as f:
            f.write(task.ground_truth_html)

        miner_html_paths = []
        for solution in solutions:
            path = f"{WORK_DIR}/miner{solution.miner_uid}_{uuid.uuid4()}.html"
            with open(path, "w") as f:
                f.write(solution.html)
            miner_html_paths.append(path)

        high_level_scores = await high_level_matching_score(miner_html_paths, original_html_path)
        low_level_scores = await low_level_matching_score(miner_html_paths, original_html_path)
        
        bt.logging.debug(f"Visual scores: {high_level_scores}")
        bt.logging.debug(f"Visual scores: {low_level_scores}")

        scores = high_level_scores * 0.3 + low_level_scores * 0.7
        await stop_browser()
        return scores
