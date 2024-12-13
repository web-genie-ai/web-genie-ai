# prompt to make rounded trip correctness
PROMPT_RTC = """
You are an HTML, CSS expert. And you are well versed in the AI, ML.
I have a model that converts the prompt to html.
I want you to analyze the html code and make a prompt that generate the given html code.
The following is the given html code:
{html}
The following is the example of prompt:
{prompt}

{instructions}
"""