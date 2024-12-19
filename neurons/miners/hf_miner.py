import bittensor as bt
import os

from webgenie.base.neuron import BaseNeuron
from webgenie.protocol import WebgenieTextSynapse, WebgenieImageSynapse
from webgenie.helpers.images import base64_to_image

from webgenie.utils.gpus import get_gpu_info
total_memory_mb, _, _ = get_gpu_info()

if total_memory_mb is None:
    raise ValueError("No GPU detected. HfMiner requires a GPU.")

bt.logging.info(f"Total memory: {total_memory_mb}")

if total_memory_mb < 1024 * 23:
    raise ValueError("Insufficient GPU memory. HfMiner requires at least 25GB of GPU memory.")

from neurons.miners.hf_models.websight_finetuned import generate_html_from_image

if total_memory_mb > 1024 * 50:
    from neurons.miners.hf_models.falcon7b import generate_html_from_text

class HfMiner:
    def __init__(self, neuron: BaseNeuron):
        self.neuron = neuron
 
    async def forward_text(self, synapse: WebgenieTextSynapse) -> WebgenieTextSynapse:  
        try:  
            if total_memory_mb > 1024 * 50:
                synapse.html = generate_html_from_text(synapse.prompt)
            else:
                synapse.html = "you don't have enough memory to generate html from text"
            return synapse
        except Exception as e:
            bt.logging.error(f"Error in HfMiner forward_text: {e}")
            synapse.html = f"Error in HfMiner forward_text: {e}"
            return synapse

    async def forward_image(self, synapse: WebgenieImageSynapse) -> WebgenieImageSynapse:
        try:
            bt.logging.debug(f"Generating HTML from image")
            synapse.html = generate_html_from_image(base64_to_image(synapse.base64_image))
            bt.logging.debug(f"Generated HTML: {synapse.html}")
            return synapse
        except Exception as e:
            bt.logging.error(f"Error in OpenaiMiner forward_image: {e}")
            synapse.html = f"Error in OpenaiMiner forward_image: {e}"
            return synapse