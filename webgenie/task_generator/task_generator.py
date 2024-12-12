import bittensor as bt
from typing import List, Tuple

from webgenie.tasks import Task
from webgenie.solution import Solution

class TaskGenerator:
    """
    A singleton generator for tasks.
    """
    def __init__(self):
        pass       

    async def generate_task(self) -> Tuple[Task, bt.Synapse]:
        pass

    async def reward(self, task: Task, responses: List[Solution]) -> List[float]:
        pass

