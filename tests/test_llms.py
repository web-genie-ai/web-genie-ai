import sys
import os
import random

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(filename=".env.validator"))

import asyncio
from pydantic import BaseModel

from webgenie.helpers.llms import openai_call, set_seed


class HTML(BaseModel):
    html: str


async def test_openai_call():
    result = await openai_call(
        messages = [
            {"role": "system", "content": "Could you make the following webpage more complex? It should be more complex and have more elements."},
            {"role": "user", "content": """<!DOCTYPE html>\n<html lang="en">\n<head>\n    <meta charset="UTF-8">\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n    <title>Red Background with Blue Rectangle</title>\n    <style>\n        body {\n            margin: 0;\n            height: 100vh;\n            display: flex;\n            justify-content: center;\n            align-items: center;\n            background-color: red;\n        }\n        .blue-rectangle {\n            width: 300px;\n            height: 200px;\n            background-color: blue;\n        }\n    </style>\n</head>\n<body>\n    <div class="blue-rectangle"></div>\n</body>\n</html>"""},
        ], 
        response_format = HTML,
    )
    print(result)


if __name__ == "__main__":
    set_seed(5)
    asyncio.run(test_openai_call())

    