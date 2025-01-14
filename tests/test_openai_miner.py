import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(filename=".env.validator"))

import asyncio

from webgenie.helpers.images import image_to_base64
from webgenie.protocol import WebgenieTextSynapse, WebgenieImageSynapse

from neurons.miners.openai_miner import OpenaiMiner


async def test_openai_miner():
    miner = OpenaiMiner(neuron = None)
    # result = await miner.forward_text(synapse = WebgenieTextSynapse(prompt = "Create a webpage with a red background and a blue rectangle in the center."))
    # print(result)

    result = await miner.forward_image(
        synapse = WebgenieImageSynapse(
            base64_image = image_to_base64("debug_images/image_20250106_225751.png"),
            prompt = "Create a webpage with a red background and a blue rectangle in the center."
        )
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(test_openai_miner())