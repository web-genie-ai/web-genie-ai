
from btcopilot.tasks import Task
class TaskGenerator:
    """
    A singleton generator for tasks.
    """
    def __init__(self):
        pass       

    def next_task(self) -> Task:
        return Task(query="CommingSoon Page with goback button, navHeader, and footer" , timeout=50)

