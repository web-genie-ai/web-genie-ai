from abc import ABC, abstractmethod
import numpy as np
from typing import List

from webgenie.tasks import Task, Solution


class Reward(ABC):
    @abstractmethod
    async def reward(self, task: Task, solutions: List[Solution]) -> np.ndarray:
        pass