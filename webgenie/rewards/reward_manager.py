from typing import Dict, List, Tuple

from webgenie.rewards import Reward
from webgenie.rewards import GPTReward
from webgenie.rewards import SpeedReward
from webgenie.rewards import IsValidReward
from webgenie.tasks import Task
from webgenie.solution import Solution

class RewardManager:
    """
    A singleton manager for the reward models.
    """
    reward_models: Dict[str, Reward] = {}

    def __init__(self):
        self.reward_models = {
            "gpt": GPTReward(),
            "speed": SpeedReward(),
            "is_valid": IsValidReward(),
        }
        
    def _penalty(self, task: Task, solution: Solution) -> float:
        """
        Penalize the solution based on the penalty models.
        """
        penalty = 0
        for penalty_model in task.penalty_models:
            model = self.reward_models[penalty_model[0]]  
            weight = penalty_model[1]
            
            penalty += weight * model.reward(task, solution)       
        return penalty

    def _reward(self, task: Task, solution: Solution) -> float:
        """
        Reward the solution based on the reward models.
        """
        reward = 0
        for reward_model in task.reward_models:
            model = self.reward_models[reward_model[0]]
            weight = reward_model[1]
            reward += weight * model.reward(task, solution)
        
        return reward
    
    def _score(self, task: Task, solution: Solution) -> float:
        """
        Score the solution based on the reward and penalty models.
        """
        score = task.reward_weight * self._reward(task, solution) - task.penalty_weight * self._penalty(task, solution)
        if score < 0:
            score = 0
        return score

    def score(self, task: Task, results: List[Solution]) -> Tuple[List[float], List[int]]:
        """
        Score the solutions based on the reward and penalty models.
        """
        scores = []
        miner_uids = []
        for solution in results:
            score = self._score(task, solution)
            scores.append(score)
            miner_uids.append(solution.miner_uid)
        
        return scores, miner_uids