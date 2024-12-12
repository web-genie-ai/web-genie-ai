from typing import Any
from pydantic import BaseModel, Field

class Task(BaseModel):
    timeout: float = Field(default=50)
    generator: Any = Field(default=None)

class ImageTask(Task):
    base64_image: str = Field(default="")

class TextTask(Task):
    prompt: str = Field(default="")
