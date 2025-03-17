# https://huggingface.co/datasets/SALT-NLP/Design2Code_human_eval_pairwise

import bittensor as bt
import random
import requests
from datasets import load_dataset

from webgenie.datasets.dataset import Dataset, DatasetEntry


class CentralDataset(Dataset):    
    HOTKEY = "hotkey"
    SIGNATURE = "signature"
    
    def __init__(self):
        pass
    
    async def generate_context(self, **kwargs)->DatasetEntry:
        try:
            bt.logging.info("Generating Central context")
            session = kwargs.get("session")
            task_number = kwargs.get("task_number")
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
        bt.logging.info(f"Getting HTML for session {session} and task {task_number}")
        method = "GET"
        url = f"http://209.126.9.130:18000/api/v1/task/generate"
        headers = {
            "Signature": CentralDataset.SIGNATURE,
            "Hotkey": CentralDataset.HOTKEY
        }
        params = {
            "session": int(session),
            "task_number": int(task_number)
        }
        response = requests.request(method, url, headers=headers, params=params)
        
        if response.status_code != 200:
            raise Exception(f"Failed to get HTML: {response.status_code} {response.text}")
        
        return response.json()["html"]
