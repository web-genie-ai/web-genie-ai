# https://huggingface.co/datasets/SALT-NLP/Design2Code_human_eval_pairwise

import bittensor as bt
import random
from datasets import load_dataset
from pydantic import BaseModel, Field

from webgenie.datasets.dataset import Dataset, DatasetEntry
from webgenie.helpers.llms import openai_call
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

    async def _make_html_complex(self, html: str)->str:
        bt.logging.info("Making HTML complex")
        response = await openai_call(
            messages = [
                {"role": "system", "content": PROMPT_MAKE_HTML_COMPLEX},
                {"role": "user", "content": html},
            ],
            response_format = HTMLResponse,
        )
        return response.complex_html

    async def generate_context(self)->DatasetEntry:
        try:
            bt.logging.info("Generating Huggingface context")
            random_index = random.randint(0, len(self.dataset) - 1)
            html = self.dataset[random_index][self.html_column]
            complex_html = await self._make_html_complex(html)
            return DatasetEntry(
                src="huggingface",
                url=f"design2code_{random_index}",
                ground_truth_html=complex_html,
                prompt="",
                base64_image=""
            )
        except Exception as e:
            bt.logging.error(f"Error in generate_context: {e}")
            raise e
            
