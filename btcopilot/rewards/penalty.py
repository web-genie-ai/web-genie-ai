from btcopilot.rewards import Reward
from btcopilot.tasks import Task
from btcopilot.solution import Solution

class PenaltyReward(Reward):

    def __init__(self):
        pass

    def reward(self, task: Task, solution: Solution) -> float:
        return 0.0