import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(filename=".env.validator"))

import asyncio
from pydantic import BaseModel

from webgenie.helpers.llms import openai_call


class HTML(BaseModel):
    html: str


async def test_openai_call():
    result = await openai_call(
        messages = [
            {"role": "system", "content": "You are an expert web developer who specializes in HTML and CSS. A user will provide you with the webpage requirements. You need to return a single html file that uses HTML and CSS to satisfy the requirements. Include all CSS code in the HTML file itself. If it involves any images, use 'rick.jpg' as the placeholder. Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions. Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout. Respond with the content of the HTML+CSS file:"},
            {"role": "user", "content": "Create a webpage with a red background and a blue rectangle in the center."},
        ], 
        response_format = HTML,
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(test_openai_call())

    