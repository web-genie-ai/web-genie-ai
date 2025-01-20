import asyncio
import sys
import os
import numpy as np
from dotenv import load_dotenv, find_dotenv
def init_test():
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(parent_dir)
    load_dotenv(find_dotenv(filename=".env.validator"))

init_test()


from webgenie.rewards.visual_reward.common.browser import start_browser, stop_browser
from webgenie.rewards.visual_reward.common.extract_html_elements import extract_html_elements
from webgenie.rewards.visual_reward.low_level_matching_score.text_matching_score import calculate_text_matching_similarity
from webgenie.rewards.visual_reward.low_level_matching_score.input_matching_score import calculate_input_matching_similarity
from webgenie.rewards.visual_reward.low_level_matching_score.element_matching_score import calculate_element_matching_similarity
from webgenie.rewards.visual_reward.high_level_matching_score.clip_matching_score import calculate_clip_score
from webgenie.rewards.visual_reward.high_level_matching_score.histogram import histogram_matching_score
from webgenie.rewards.visual_reward.high_level_matching_score.high_level_matching_score import high_level_matching_score

async def test_text_matching_score():
    await start_browser()
    import time
    start_time = time.time()
    url = "test1.html"
    url_predict = "miner.html"
    scores = await high_level_matching_score([url_predict], url)
    print(scores)
    print(time.time() - start_time)
    await stop_browser()

if __name__ == "__main__":
    asyncio.run(test_text_matching_score())


