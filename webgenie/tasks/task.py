from typing import Any
from pydantic import BaseModel, Field

class Task(BaseModel):
    timeout: float = Field(default=50)
    competition: Any = Field(default=None)

class ImageTask(Task):
    base64_image: str = Field(default="", description="The base64 encoded image")
    ground_truth_html: str = Field(default="", description="The ground truth html")

class TextTask(Task):
    prompt: str = Field(default="", description="The prompt for the text task")
    ground_truth_html: str = Field(default="", description="The ground truth html")
