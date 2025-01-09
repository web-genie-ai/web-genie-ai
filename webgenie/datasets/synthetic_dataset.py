# The paper [Unlocking the conversion of Web Screenshots into HTML Code with the WebSight Dataset]
# (https://arxiv.org/pdf/2403.09029v1#bib.bib5) is our inspiration.
# The paper suggests using Mistral-7B-Instruct to generate concepts and use Deepseek-Coder-33b-instruct 
# to generate html, but now we are using openai models here. We are going to use that models on the mainnet

import bittensor as bt
from typing import List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from webgenie.datasets.dataset import Dataset, DatasetEntry
from webgenie.helpers.llms import call_llm
from webgenie.prompts import PROMPT_GEN_CONCEPT, PROMPT_GEN_HTML


class ConceptResponse(BaseModel):
    concepts: List[str] = Field(description="The concept of the website")


class HTMLResponse(BaseModel):
    html: str = Field(description="The html code of the website")


class SyntheticDataset(Dataset):
    def __init__(self, has_ground_truth_html: bool = True):
        self.has_ground_truth_html = has_ground_truth_html
        self.concept_parser = JsonOutputParser(pydantic_object=ConceptResponse)
        self.html_parser = JsonOutputParser(pydantic_object=HTMLResponse)
        self.concepts = []

    async def _generate_concepts(self):
        bt.logging.info("Generating concepts")
        response = await call_llm(
            template=[
                ("system", PROMPT_GEN_CONCEPT),
            ],
            params={"instructions": self.concept_parser.get_format_instructions()},
            output_parser=self.concept_parser,
        )
        return response["concepts"]

    async def _generate_html(self, concept: str):
        bt.logging.info("Generating HTML from concept")
        response = await call_llm(
            template=[
                ("system", PROMPT_GEN_HTML),
            ],
            params={
                "concept": concept, 
                "instructions": self.html_parser.get_format_instructions(),
            },
            output_parser=self.html_parser,
        )
        return response["html"]
        
    async def generate_context(self)->DatasetEntry:
        bt.logging.info("Generating Synthetic context")
        if not self.concepts:
            self.concepts = await self._generate_concepts()
        
        concept = self.concepts.pop(0)
        
        if self.has_ground_truth_html == True:
            ground_truth_html = await self._generate_html(concept)
        else:
            ground_truth_html = ""

        return DatasetEntry(
            src="synthetic",
            prompt=concept,
            ground_truth_html=ground_truth_html,
        )
