# The paper [Design2Code: Benchmarking Multimodal Code Generation for Automated Front-End Engineering]
# (https://arxiv.org/pdf/2403.03163) is our inspiration for this reward.

import bittensor as bt
import os
import asyncio
import multiprocessing
import numpy as np
import shutil
import uuid
from datetime import datetime
from typing import List

from webgenie.constants import WORK_DIR
from webgenie.rewards.reward import Reward
from webgenie.rewards.visual_reward.common.browser import start_browser, stop_browser
from webgenie.rewards.visual_reward.high_level_matching_score import high_level_matching_score
from webgenie.rewards.visual_reward.low_level_matching_score import low_level_matching_score
from webgenie.tasks import Task, ImageTask, Solution


class VisualReward(Reward):
    def __init__(self):
        pass

    async def reward_worker(self, task: Task, solutions: List[Solution], current_work_dir: str) -> np.ndarray:
        await start_browser()
        
        original_html_path = f"{current_work_dir}/original_{uuid.uuid4()}.html"
        with open(original_html_path, "w") as f:
            f.write(task.ground_truth_html)

        miner_html_paths = []
        for solution in solutions:
            path = f"{current_work_dir}/miner{solution.miner_uid}_{uuid.uuid4()}.html"
            with open(path, "w") as f:
                f.write(solution.html)
            miner_html_paths.append(path)
        try:
            high_level_scores = await high_level_matching_score(miner_html_paths, original_html_path)
        except Exception as e:
            bt.logging.error(f"Error in high_level_matching_score: {e}")
            high_level_scores = np.zeros(len(miner_html_paths))
        try:
            low_level_scores = await low_level_matching_score(miner_html_paths, original_html_path)
        except Exception as e:
            bt.logging.error(f"Error in low_level_matching_score: {e}")
            low_level_scores = np.zeros(len(miner_html_paths))
        
        bt.logging.debug(f"High level visual scores: {high_level_scores}")
        bt.logging.debug(f"Low level visual scores: {low_level_scores}")

        scores = high_level_scores * 0.3 + low_level_scores * 0.7
        await stop_browser()
        return scores
    
    def sync_reward_worker(self, task: Task, solutions: List[Solution], current_work_dir: str) -> np.ndarray:
        try:
            # Timeout of 2 hours for visual reward processing
            VISUAL_REWARD_TIMEOUT = 60 * 60 * 2 # 2 hours
            
            # Run the async reward worker with timeout
            return asyncio.run(
                asyncio.wait_for(
                    self.reward_worker(task, solutions, current_work_dir),
                    timeout=VISUAL_REWARD_TIMEOUT
                )
            )
        except Exception as e:
            bt.logging.error(f"Error in sync_reward_worker: {e}")
            return [0] * len(solutions)

    async def reward(self, task: Task, solutions: List[Solution]) -> np.ndarray:
        if not isinstance(task, ImageTask):
            raise ValueError(f"Task is not a ImageTask: {type(task)}")

        bt.logging.info(f"Rewarding image task in visual reward")

        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        current_work_dir = f"{WORK_DIR}/task_{timestamp}_{task.task_id}"
        os.makedirs(current_work_dir, exist_ok=True)

        bt.logging.info(f"The number of cpu cores: {os.cpu_count()}")
        # Use ProcessPoolExecutor for parallel processing
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            # Convert solutions into chunks for parallel processing
            chunk_size = max(1, len(solutions) // os.cpu_count())
            solution_chunks = [solutions[i:i + chunk_size] for i in range(0, len(solutions), chunk_size)]
            
            # Create partial tasks for each chunk
            futures = []
            for chunk in solution_chunks:
                future = pool.apply_async(self.sync_reward_worker, args=(task, chunk, current_work_dir))
                futures.append(future)
            
            # Gather all results
            chunk_scores = []
            for future in futures:
                chunk_scores.extend(future.get())
                
            scores = np.array(chunk_scores)
        # Clean up work directory and its contents
        try:
            shutil.rmtree(current_work_dir)
        except Exception as e:
            bt.logging.warning(f"Error cleaning up work directory: {e}")
        
        return scores
