import bittensor as bt
from typing import List, Tuple

from webgenie.protocol import WebgenieImageSynapse
from webgenie.solution import Solution
from webgenie.tasks.task import Task, ImageTask
from webgenie.task_generator.task_generator import TaskGenerator

class ImageTaskGenerator(TaskGenerator):
    def __init__(self):
        super().__init__()

    async def generate_task(self) -> Tuple[Task, bt.Synapse]:
        return ImageTask(
            base64_image="base64_image" , 
            timeout=50,
            generator=self
        ), WebgenieImageSynapse(base64_image="base64_image")

    async def reward(self, task: Task, responses: List[Solution]) -> List[float]:
        pass