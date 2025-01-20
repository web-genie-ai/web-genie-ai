import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(filename=".env.validator"))

import asyncio

from webgenie.datasets.huggingface_dataset import HuggingfaceDataset


async def test_huggingface_dataset():
    dataset = HuggingfaceDataset(dataset_name="SALT-NLP/Design2Code-hf", split="train", html_column="text")
    print(await dataset.generate_context())


if __name__ == "__main__":
    asyncio.run(test_huggingface_dataset())