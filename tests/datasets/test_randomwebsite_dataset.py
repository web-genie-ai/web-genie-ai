import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(filename=".env.validator"))

import asyncio

from webgenie.datasets.random_website_dataset import RandomWebsiteDataset

async def test_random_website_dataset():
    dataset = RandomWebsiteDataset()
    await dataset.generate_context()

if __name__ == "__main__":
    asyncio.run(test_random_website_dataset())