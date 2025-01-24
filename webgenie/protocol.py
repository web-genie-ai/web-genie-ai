# The MIT License (MIT)
# Copyright © 2024 pycorn

import bittensor as bt
import hashlib
import pydantic
import random


class WebgenieTextSynapse(bt.Synapse):
    """
    A protocol for the webgenie text task.
    """
    prompt: str = pydantic.Field(
        "",
        title="Prompt",
        description="The prompt to be sent to miners.",
    )

    competition_type: str = pydantic.Field(
        "",
        title="Competition Type",
        description="The competition type.",
    )

    html: str = pydantic.Field(
        "",
        title="HTML",
        description="The HTML received from miners.",
    )


class WebgenieImageSynapse(bt.Synapse):
    """
    A protocol for the webgenie image task.
    """
    VERSION: str = pydantic.Field(
        "NONE",
        title="Version",
        description="The version of the protocol.",
    )

    task_id: str = pydantic.Field(
        "",
        title="Task ID",
        description="The task ID.",
    )

    base64_image: str = pydantic.Field(
        "",
        title="Base64 Image",
        description="The base64 image to be sent to miners.",
    )
    
    competition_type: str = pydantic.Field(
        "",
        title="Competition Type",
        description="The competition type.",
    )

    html: str = pydantic.Field(
        "",
        title="HTML",
        description="The HTML received from miners.",
    )

    html_hash: str = pydantic.Field(
        "",
        title="HTML Hash",
        description="The hash of the HTML.",
    )

    nonce: int = pydantic.Field(
        0,
        title="Nonce",
        description="The nonce.",
    )


def add_answer_hash(self, uid: int, html: str) -> int:
    nonce = uid
    hash_input = html + str(nonce)
    self.html_hash = hashlib.sha256(hash_input.encode()).hexdigest()
    self.nonce = nonce
    return nonce


def verify_answer_hash(self) -> bool:
    hash_input = self.html + str(self.nonce)
    return hashlib.sha256(hash_input.encode()).hexdigest() == self.html_hash


def hide_secret_info(self):
    self.html = ""