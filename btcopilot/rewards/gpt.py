import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
import bittensor as bt
from btcopilot.rewards import Reward
from btcopilot.tasks import Task
from btcopilot.solution import Solution

class GPTReward(Reward):
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        self.model = ChatOpenAI(
            api_key=api_key,
            model="gpt-4",    
        )
    def reward(self, task: Task, solution: Solution) -> float: 
        # Use GPT to evaluate the code
        query = f"""
        Task: {task.query}
        Generated Code:
        "```CSS"
        {solution.css}
        "```HTML"
        {solution.html}
        "```"
        Evaluate the generated code based on the following criteria:
        1. Correctness: Does the code implement the required functionality?
        2. Code quality: Is the code well-structured and following best practices?
        3. Completeness: Does the code address all aspects of the task?

        Provide a score between 0 and 1, where 1 is perfect and 0 is completely incorrect.
        Only return the score, without any explanations.
        """

        try:
            messages=[
                    {"role": "system", "content": "You are a code evaluation expert."},
                    {"role": "user", "content": query}
                ]   
            evaluation = self.model.invoke(messages)
            bt.logging.info(f"Evaluation: {evaluation.content}")
            return float(evaluation.content)
        except Exception as e:
            bt.logging.error(f"Error evaluating code: {e}")
            return 0.0