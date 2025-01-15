import asyncio
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(filename=".env.validator"))

from webgenie.rewards.lighthouse_reward import LighthouseReward
from webgenie.tasks import Task, Solution

async def test_lighthouse_reward():
    reward = LighthouseReward()
    task = Task(
        prompt="Create a website that displays a list of 10 random numbers.",
        ground_truth_html="",
    )
    html = "<html><body><h1>Hello, World!</h1></body></html>"
    html2 = "<html><body><h1>Hello, World!</h1><article>Hello, World!</article></body></html>"
    solutions = [Solution(html=html) for _ in range(5)] + [Solution(html=html2) for _ in range(5)]
    print(await reward.reward(task, solutions))

if __name__ == "__main__":
    asyncio.run(test_lighthouse_reward())
