import bittensor as bt
from typing import List, Tuple

from webgenie.protocol import WebgenieTextSynapse
from webgenie.solution import Solution
from webgenie.tasks.task import Task, TextTask
from webgenie.task_generator.task_generator import TaskGenerator

class TextTaskGenerator(TaskGenerator):
    def __init__(self):
        super().__init__()

    async def generate_task(self) -> Tuple[Task, bt.Synapse]:
        return TextTask(
            prompt="CommingSoon Page with goback button, navHeader, and footer" , 
            timeout=50,
            generator=self
        ), WebgenieTextSynapse(prompt="CommingSoon Page with goback button, navHeader, and footer")

    async def reward(self, task: Task, solutions: List[Solution]) -> List[float]:
        if not isinstance(task, TextTask):
            raise ValueError(f"Task is not a TextTask: {type(task)}")
        bt.logging.debug(task.prompt)
        bt.logging.debug(f"Rewarding text task {task} with solutions {solutions}")

        return [1.0] * len(solutions)
