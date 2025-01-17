# The paper [Unlocking the conversion of Web Screenshots into HTML Code with the WebSight Dataset]
# (https://arxiv.org/pdf/2403.09029v1#bib.bib5) is our inspiration.
# The paper suggests using Mistral-7B-Instruct to generate concepts and use Deepseek-Coder-33b-instruct 
# to generate html, but now we are using openai models here. We are going to use that models on the mainnet

import bittensor as bt
from typing import List
from pydantic import BaseModel, Field

from webgenie.datasets.dataset import Dataset, DatasetEntry
from webgenie.helpers.llms import openai_call
from webgenie.prompts import PROMPT_GEN_CONCEPT, PROMPT_GEN_HTML


class ConceptResponse(BaseModel):
    concepts: List[str] = Field(description="The concept of the website")


class HTMLResponse(BaseModel):
    html: str = Field(description="The html code of the website")


class SyntheticDataset(Dataset):
    def __init__(self, has_ground_truth_html: bool = True):
        self.has_ground_truth_html = has_ground_truth_html
        self.concepts = []

    async def _generate_concepts(self):
        bt.logging.info("Generating concepts")
        response = await openai_call(
            messages = [
                {"role": "system", "content": PROMPT_GEN_CONCEPT},
            ],
            response_format = ConceptResponse,
        )
        return response.concepts

    async def _generate_html(self, concept: str):
        bt.logging.debug(f"Generating HTML from concept: {concept}")
        response = await openai_call(
            messages = [
                {"role": "system", "content": PROMPT_GEN_HTML.format(concept=concept)},
            ],
            response_format = HTMLResponse,
        )
        return response.html
        
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
