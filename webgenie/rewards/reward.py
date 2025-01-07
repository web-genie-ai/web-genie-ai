from abc import ABC, abstractmethod
import numpy as np
from typing import List

from webgenie.tasks import Task
from webgenie.tasks.solution import Solution


class Reward(ABC):
    @abstractmethod
    async def reward(self, task: Task, solutions: List[Solution]) -> np.ndarray:
        pass