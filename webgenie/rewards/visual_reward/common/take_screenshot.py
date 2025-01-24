import bittensor as bt
import os
from PIL import Image

from webgenie.constants import (
    DEFAULT_LOAD_TIME, 
    CHROME_HTML_LOAD_TIME, 
    JAVASCRIPT_RUNNING_TIME,
)
from webgenie.rewards.visual_reward.common.browser import web_player


async def take_screenshot(url, output_file_path, load_time = DEFAULT_LOAD_TIME, overwrite = False):
    if os.path.exists(url):
        url = f"file:///{os.path.abspath(url)}"

    if os.path.exists(output_file_path):
        if not overwrite:
            return
        else:
            os.remove(output_file_path)
        
    try:
        page = await web_player["browser"].new_page()
        await page.goto(url, timeout=CHROME_HTML_LOAD_TIME)

        await page.wait_for_load_state('networkidle')
        await page.wait_for_timeout(JAVASCRIPT_RUNNING_TIME)
        
        await page.screenshot(
            path=output_file_path, 
            full_page=True, 
            animations='disabled', 
            timeout=CHROME_HTML_LOAD_TIME,
        )
        await page.close()
    except Exception as e:
        bt.logging.error(f"Failed to take screenshot due to: {e}. Generating a blank image.")
        # Generate a blank image 
        img = Image.new('RGB', (1280, 960), color = 'white')
        img.save(output_file_path)
    