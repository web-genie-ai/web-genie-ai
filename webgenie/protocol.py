# The MIT License (MIT)
# Copyright © 2024 pycorn

import bittensor as bt
import pydantic
import json
from typing import AsyncIterator, Union, Any
from starlette.responses import StreamingResponse

from webgenie.solution import Solution
from webgenie.tasks import Task

class WebgenieTextSynapse(bt.Synapse):
    """
    A protocol for the webgenie text task.
    """
    prompt: str = pydantic.Field(
        "",
        title="Prompt",
        description="The prompt to be sent to miners."
    )

    solution: Union[Solution, None] = pydantic.Field(
        None,
        title="Solution",
        description="A solution received from miners."
    )

class WebgenieImageSynapse(bt.Synapse):
    """
    A protocol for the webgenie image task.
    """
    base64_image: str = pydantic.Field(
        "",
        title="Base64 Image",
        description="The base64 image to be sent to miners."
    )

    solution: Union[Solution, None] = pydantic.Field(
        None,
        title="Solution",
        description="A solution received from miners."
    )


class WebgenieStreamingSynapse(bt.StreamingSynapse):
    """
    A protocol for the webgenie streaming task.
    """

    task: Union[Task, None] = pydantic.Field(
        None,
        title="Task",
        description="A task to be sent to miners."
    )

    solution: Union[Solution, None] = pydantic.Field(
        None,
        title="Solution",
        description="A solution received from miners."
    )

    completion: str = pydantic.Field(
        "",
        title="Completion",
        description="The completion response from miners."
    )

    async def process_streaming_response(self, response: StreamingResponse) -> AsyncIterator[str]:
        """
        Processes a streaming response from a miner.
        """
        if self.completion is None:
            self.completion = ""
        async for chunk in response.content.iter_any():
            tokens = chunk.decode("utf-8") 
            
            for token in tokens:
                if token:
                    self.completion += token
            yield tokens

    def deserialize(self) -> Union[Any, None]:
        """
        Deserializes the response.
        """
        try:
            json_response = json.loads(self.completion)
            css = None
            html = None
            for key, value in json_response.items():
                if key.lower() == "css":    
                    css = value
                elif key.lower() == "html":
                    html = value

            if css is None and html is None:
                bt.logging.error(f"Invalid response: {json_response}")
                return None
            
            css = str(css)
            html = str(html)

            process_time = self.dendrite.process_time
            bt.logging.debug(f"css: {css}, html: {html}, process_time: {process_time}")
            self.solution = Solution(css=css, html=html, process_time=process_time, miner_uid=0)
            return self 
        except Exception as e:
            bt.logging.error(f"Failed to parse completion: {e}")
            return None
        
    
    def extract_response_json(self, response: StreamingResponse) -> dict:
        """
        Extracts the response JSON.
        """
        headers = {
            k.decode("utf-8"): v.decode("utf-8")
            for k, v in response.__dict__["_raw_headers"]
        }

        def extract_info(prefix:str) -> dict:
            return {
                key.split("_")[-1]: value
                for key, value in headers.items()
                if key.startswith(prefix)
            }
        return {
            "completion": self.completion
        }
    