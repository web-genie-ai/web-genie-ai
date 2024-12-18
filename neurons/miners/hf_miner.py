import bittensor as bt
import os

from webgenie.base.neuron import BaseNeuron
from webgenie.protocol import WebgenieTextSynapse, WebgenieImageSynapse
from webgenie.helpers.images import base64_to_image

from neurons.miners.hf_models.websight_finetuned import generate_html_from_image

class HfMiner:
    def __init__(self, neuron: BaseNeuron):
        self.neuron = neuron

    async def forward_text(self, synapse: WebgenieTextSynapse) -> WebgenieTextSynapse:  
        try:  
            synapse.html = "dummy text response"
            return synapse
        except Exception as e:
            bt.logging.error(f"Error in OpenaiMiner forward_text: {e}")
            synapse.html = f"Error in OpenaiMiner forward_text: {e}"
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