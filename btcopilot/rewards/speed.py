
from btcopilot.rewards import Reward
from btcopilot.tasks import Task
from btcopilot.solution import Solution

class SpeedReward(Reward):

    def __init__(self):
        pass

    def reward(self, task: Task, solution: Solution) -> float:
        if (task.timeout == 0):
            return 1.0
        return 1.0 - (solution.process_time / task.timeout)
 