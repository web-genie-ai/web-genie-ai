import bittensor as bt
from pydantic import BaseModel, Field

from webgenie.base.neuron import BaseNeuron
from webgenie.helpers.llms import openai_call
from webgenie.protocol import WebgenieTextSynapse, WebgenieImageSynapse


class HTML(BaseModel):
    html: str


class OpenaiMiner:
    def __init__(self, neuron: BaseNeuron):
        self.neuron = neuron

    async def forward_text(self, synapse: WebgenieTextSynapse) -> WebgenieTextSynapse:  
        try:  
            html_response = await openai_call(
                messages = [
                    {"role": "system", "content": "You are an expert web developer who specializes in HTML and CSS. A user will provide you with the webpage requirements. You need to return a single html file that uses HTML and CSS to satisfy the requirements. Include all CSS code in the HTML file itself. If it involves any images, use 'rick.jpg' as the placeholder. Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions. Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout. Respond with the content of the HTML+CSS file:"},
                    {"role": "user", "content": "Create a webpage with a red background and a blue rectangle in the center."},
                ], 
                response_format = HTML,
            )
            synapse.html = html_response.html
            return synapse
        except Exception as e:
            bt.logging.error(f"Error in OpenaiMiner forward_text: {e}")
            synapse.html = f"Error in OpenaiMiner forward_text: {e}"
            return synapse

    async def forward_image(self, synapse: WebgenieImageSynapse) -> WebgenieImageSynapse:
        try:
            html_response = await openai_call(
                messages =  [
                    {
                        "role": "system", 
                        "content": """You are an expert web developer who specializes in HTML and CSS.
                            A user will provide you with a screenshot of a webpage, along with all texts that they want to put on the webpage.  
                            You need to return a single html file that uses HTML and CSS to reproduce the given website.   
                            Include all CSS code in the HTML file itself.
                            If it involves any images, use "rick.jpg" as the placeholder.
                            Some images on the webpage are replaced with a blue rectangle as the placeholder, use "rick.jpg" for those as well.       
                            Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions.
                            Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.
                            Respond with the content of the HTML+CSS file:
                        """
                    },
                    {
                        "role": "user",
                        "content": [{
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{synapse.base64_image}",
                            },
                        }],
                    },
                ],
                response_format = HTML,
            )

            synapse.html = html_response.html
            return synapse
        except Exception as e:
            bt.logging.error(f"Error in OpenaiMiner forward_image: {e}")
            synapse.html = f"Error in OpenaiMiner forward_image: {e}"
            return synapse