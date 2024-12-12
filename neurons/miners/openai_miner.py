import bittensor as bt
import os

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from webgenie.base.neuron import BaseNeuron
from webgenie.protocol import WebgenieTextSynapse, WebgenieImageSynapse
from webgenie.solution.solution import Solution

class HTMLResponse(BaseModel):
    html: str = Field(default="", description="The HTML code for the webpage")

class OpenaiMiner:
    def __init__(self, neuron: BaseNeuron):
        self.neuron = neuron

        self.model = ChatOpenAI(
            api_key= os.getenv("OPENAI_API_KEY"),
            model_name="gpt-4o",
        )

        self.html_response_parser = JsonOutputParser(pydantic_object=HTMLResponse)

    async def forward_text(self, synapse: WebgenieTextSynapse) -> WebgenieTextSynapse:  
        try:  
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert web developer who specializes in HTML and CSS. A user will provide you with the webpage requirements. You need to return a single html file that uses HTML and CSS to satisfy the requirements. 
                Include all CSS code in the HTML file itself.
                If it involves any images, use "rick.jpg" as the placeholder.
                Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions.
                Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.
                Respond with the content of the HTML+CSS file:
                {instructions}"""),
                ("user", "{query}"),
            ])

            chain = prompt | self.model | self.html_response_parser
            html_response = chain.invoke({
                "query": synapse.prompt, 
                "instructions": self.html_response_parser.get_format_instructions()
            })
            
            synapse.solution = Solution(html=html_response["html"])
            return synapse
        except:
            bt.logging.error(f"Error in OpenaiMiner forward_text: {e}")
            return synapse

    async def forward_image(self, synapse: WebgenieImageSynapse) -> WebgenieImageSynapse:
        try:
            
            prompt_messages = [
                SystemMessagePromptTemplate.from_template("""
                You are an expert web developer who specializes in HTML and CSS.
                A user will provide you with a screenshot of a webpage, along with all texts that they want to put on the webpage.  
                You need to return a single html file that uses HTML and CSS to reproduce the given website.   
                Include all CSS code in the HTML file itself.
                If it involves any images, use "rick.jpg" as the placeholder.
                Some images on the webpage are replaced with a blue rectangle as the placeholder, use "rick.jpg" for those as well.       
                Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions.
                Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.
                Respond with the content of the HTML+CSS file:
                {instructions}"""),
                HumanMessagePromptTemplate.from_template(
                    template=[
                        {"type": "image_url", "image_url": {"url": "{image_url}"}},
                    ]
                )
            ]

            prompt = ChatPromptTemplate(messages=prompt_messages)

            chain = prompt | self.model | self.html_response_parser

            html_response = chain.invoke({
                "instructions": self.html_response_parser.get_format_instructions(),
                "image_url": f"data:image/jpeg;base64,{synapse.base64_image}",
            })

            synapse.solution = Solution(html = html_response["html"])
            return synapse
        except Exception as e:
            bt.logging.error(f"Error in OpenaiMiner forward_image: {e}")
            return synapse