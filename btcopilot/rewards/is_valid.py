from bs4 import BeautifulSoup
import tinycss2

from btcopilot.rewards import Reward
from btcopilot.tasks import Task
from btcopilot.solution import Solution

class IsValidReward(Reward):

    def __init__(self):
        pass

    def is_valid_css(self, css: str) -> bool:
  
        try:
            # Parse the CSS using tinycss2
            parsed_css = tinycss2.parse_stylesheet(css)            
            # Check if the CSS was parsed successfully
            if parsed_css is not None:
                return True
            else:
                return False
        except Exception:
            # If parsing fails, the CSS is invalid
            return False
        return True
    
    def is_valid_html(self, html: str) -> bool:
        
        try:
            # Parse the HTML using BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Check if the HTML has a valid structure
            if soup.find():
                return True
            else:
                return False
        except Exception:
            # If parsing fails, the HTML is invalid
            return False

    def reward(self, task: Task, solution: Solution) -> float:
        if self.is_valid_css(solution.css) and self.is_valid_html(solution.html):
            return 0.0
        else:
            return 1.0
