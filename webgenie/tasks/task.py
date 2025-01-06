from typing import Any
from pydantic import BaseModel, Field
from webgenie.constants import IMAGE_TASK_REWARD, TEXT_TASK_REWARD

class Task(BaseModel):
    timeout: float = Field(default=50)
    generator: Any = Field(default=None)
    reserved_reward: float = Field(default=0.0)

class ImageTask(Task):
    base64_image: str = Field(default="", description="The base64 encoded image")
    ground_truth_html: str = Field(default="", description="The ground truth html")
    reserved_reward: float = Field(default=IMAGE_TASK_REWARD)

class TextTask(Task):
    prompt: str = Field(default="", description="The prompt for the text task")
    ground_truth_html: str = Field(default="", description="The ground truth html")
    reserved_reward: float = Field(default=TEXT_TASK_REWARD)