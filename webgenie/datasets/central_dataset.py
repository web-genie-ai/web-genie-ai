# https://huggingface.co/datasets/SALT-NLP/Design2Code_human_eval_pairwise

import bittensor as bt
import random
import requests
from datasets import load_dataset

from webgenie.datasets.dataset import Dataset, DatasetEntry


class CentralDataset(Dataset):
    def __init__(self):
        pass

    async def generate_context(self) -> DatasetEntry:
        pass
    
    async def generate_context(self, session:int, task_number:int)->DatasetEntry:
        try:
            bt.logging.info("Generating Central context")
            html = self.get_html(session, task_number)
            return DatasetEntry(
                src="central",
                url=f"central_{session}_{task_number}",
                ground_truth_html=html,
                prompt="",
                base64_image=""
            )
        except Exception as e:
            bt.logging.error(f"Error in generate_context: {e}")
            raise e
            
    def get_html(self, session:int, task_number:int)->str:
        method = "GET"
        url = f"http://209.126.9.130:18000/api/v1/task/generate?session={session}&task_number={task_number}"
        response = requests.request(method, url)
        if response.status_code != 200:
            raise Exception(f"Failed to get HTML: {response.status_code} {response.text}")
        
        return response.json()["html"]

