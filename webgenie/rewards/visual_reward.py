# The paper [Design2Code: Benchmarking Multimodal Code Generation for Automated Front-End Engineering]
# (https://arxiv.org/pdf/2403.03163) is our inspiration for this reward.

import bittensor as bt
import numpy as np
from typing import List

from webgenie.constants import WORK_DIR
from webgenie.rewards.reward import Reward
from webgenie.rewards.metrics.visual_score import visual_eval_v3_multi
from webgenie.tasks.task import Task, ImageTask
from webgenie.tasks.solution import Solution


class VisualReward(Reward):
    def __init__(self):
        pass

    async def reward(self, task: Task, solutions: List[Solution]) -> np.ndarray:
        if not isinstance(task, ImageTask):
            raise ValueError(f"Task is not a ImageTask: {type(task)}")
        
        bt.logging.debug(f"Rewarding image task in visual reward")
        
        original_html_path = f"{WORK_DIR}/original.html"
        with open(original_html_path, "w") as f:
            f.write(task.ground_truth_html)

        miner_html_paths = []
        for solution in solutions:
            path = f"{WORK_DIR}/miner{solution.miner_uid}.html"
            with open(path, "w") as f:
                f.write(solution.html)
            miner_html_paths.append(path)

        visual_scores = visual_eval_v3_multi([miner_html_paths, original_html_path])
        bt.logging.debug(f"Visual scores: {visual_scores}")
        return np.array([score[1] for score in visual_scores])
