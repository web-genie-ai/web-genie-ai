# https://huggingface.co/datasets/SALT-NLP/Design2Code_human_eval_pairwise

import bittensor as bt
import os
import random
from datasets import load_dataset

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from webgenie.datasets.dataset import Dataset, DatasetEntry
from webgenie.prompts import PROMPT_MAKE_HTML_COMPLEX


class HTMLResponse(BaseModel):
    complex_html: str = Field(description="the complex html code")

class HuggingfaceDataset(Dataset):
    def __init__(self , dataset_name: str, split: str, html_field: str):
        self.dataset = load_dataset(dataset_name, split=split)
        self.html_field = html_field
        self.model = ChatOpenAI(
            base_url=os.getenv("LLM_MODEL_URL"),
            model=os.getenv("LLM_MODEL_ID"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.output_parser = JsonOutputParser(pydantic_object=HTMLResponse)

    async def _make_html_complex(self, html: str)->str:
        bt.logging.info("Making HTML complex")
        prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_MAKE_HTML_COMPLEX),
        ])
        chain = prompt | self.model | self.output_parser
        response = await chain.ainvoke({
            "html": html, 
            "instructions": self.output_parser.get_format_instructions()
        })
        return response["complex_html"]

    async def generate_context(self)->DatasetEntry:
        try:
            bt.logging.info("Generating Huggingface context")
            random_index = random.randint(0, len(self.dataset) - 1)
            html = self.dataset[random_index][self.html_field]
            complex_html = await self._make_html_complex(html)
            return DatasetEntry(
                src="huggingface",
                topic="design2code",
                ground_truth_html=complex_html,
                prompt="",
                base64_image=""
            )
        except Exception as e:
            bt.logging.error(f"Error in generate_context: {e}")
            raise e
            
