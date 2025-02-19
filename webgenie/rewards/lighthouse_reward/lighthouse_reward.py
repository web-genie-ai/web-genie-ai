# The paper [Unsupervised Evaluation of Code LLMs with Round-Trip Correctness]
# (https://arxiv.org/pdf/2402.08699#page=11&zoom=100,384,458) is our inspiration for this reward.

import bittensor as bt
import os
import multiprocessing
import numpy as np
from typing import List

from webgenie.constants import NUMBER_OF_CONCURRENT_WORKERS
from webgenie.rewards.reward import Reward
from webgenie.tasks import Task, Solution
from .get_lighthouse_score import get_lighthouse_score


class LighthouseReward(Reward):
    def __init__(self):
        pass

    def sync_reward_worker(self, htmls: List[str]) -> List[float]:
        try:
            scores_dict = get_lighthouse_score(htmls)
            scores = []
            weights = [0, 0.25, 0.25, 0.5]
            for score_dict in scores_dict:
                score = (
                    score_dict['performance'] * weights[0] + 
                    score_dict['accessibility'] * weights[1] + 
                    score_dict['best-practices'] * weights[2] + 
                    score_dict['seo'] * weights[3]
                )
                scores.append(score)
            return scores
        except Exception as e:
            bt.logging.error(f"Error getting lighthouse score: {e}")
            return [0] * len(htmls)

    async def reward(self, task: Task, solutions: List[Solution]) -> np.ndarray:
        bt.logging.info(f"Rewarding lighthouse task")
        htmls = [solution.html for solution in solutions]
        
        # Use ProcessPoolExecutor for parallel processing
        with multiprocessing.Pool(processes=NUMBER_OF_CONCURRENT_WORKERS) as pool:
            # Convert solutions into chunks for parallel processing
            chunk_size = max(1, len(htmls) // NUMBER_OF_CONCURRENT_WORKERS) 

            html_chunks = [htmls[i:i + chunk_size] for i in range(0, len(htmls), chunk_size)]
            
            # Create partial tasks for each chunk
            futures = []
            for chunk in html_chunks:
                future = pool.apply_async(self.sync_reward_worker, args=(chunk,))
                futures.append(future)
            
            # Gather all results
            scores = []
            for future in futures:
                scores.extend(future.get())
        return np.array(scores)
    