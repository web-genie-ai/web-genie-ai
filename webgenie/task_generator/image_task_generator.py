import bittensor as bt
from typing import List, Tuple

from webgenie.helpers.images import image_to_base64
from webgenie.protocol import WebgenieImageSynapse
from webgenie.solution import Solution
from webgenie.tasks.task import Task, ImageTask
from webgenie.task_generator.task_generator import TaskGenerator

class ImageTaskGenerator(TaskGenerator):
    def __init__(self):
        super().__init__()

    async def generate_task(self) -> Tuple[Task, bt.Synapse]:
        base64_image = image_to_base64("original.jpg")
        return ImageTask(
            base64_image=base64_image, 
            timeout=50,
            generator=self
        ), WebgenieImageSynapse(base64_image=base64_image)

    async def reward(self, task: Task, solutions: List[Solution]) -> List[float]:
        if not isinstance(task, ImageTask):
            raise ValueError(f"Task is not a ImageTask: {type(task)}")
        bt.logging.debug(task.base64_image)
        bt.logging.debug(f"Rewarding image task {task} with solutions {solutions}")
        return [1.0] * len(solutions)
