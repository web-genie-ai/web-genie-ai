# https://huggingface.co/datasets/SALT-NLP/Design2Code_human_eval_pairwise

import bittensor as bt
import random
from datasets import load_dataset

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from webgenie.datasets.dataset import Dataset, DatasetEntry
from webgenie.helpers.llms import call_llm
from webgenie.prompts import PROMPT_MAKE_HTML_COMPLEX


class HTMLResponse(BaseModel):
    complex_html: str = Field(description="the complex html code")


class HuggingfaceDataset(Dataset):
    def __init__(self , **kwargs):
        dataset_name = kwargs["dataset_name"]
        html_column = kwargs["html_column"]
        split = kwargs["split"]

        self.dataset = load_dataset(dataset_name, split=split)
        self.html_column = html_column
        self.output_parser = JsonOutputParser(pydantic_object=HTMLResponse)

    async def _make_html_complex(self, html: str)->str:
        bt.logging.info("Making HTML complex")
        response = await call_llm(
            template=[
                ("system", PROMPT_MAKE_HTML_COMPLEX),
            ],
            params={"html": html, "instructions": self.output_parser.get_format_instructions()},
            output_parser=self.output_parser
        )
        return response["complex_html"]

    async def generate_context(self)->DatasetEntry:
        try:
            bt.logging.info("Generating Huggingface context")
            random_index = random.randint(0, len(self.dataset) - 1)
            html = self.dataset[random_index][self.html_column]
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
            
