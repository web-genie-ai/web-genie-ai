import os
from PIL import Image

from webgenie.constants import DEFAULT_LOAD_TIME
from webgenie.rewards.visual_reward.common.browser import web_player


async def take_screenshot(url, output_file_path, load_time = DEFAULT_LOAD_TIME, overwrite = False):
    if os.path.exists(url):
        url = f"file:///{os.path.abspath(url)}"
    if os.path.exists(output_file_path) and not overwrite:
        return
    if os.path.exists(output_file_path) and overwrite:
        os.remove(output_file_path)
    print(f"Taking screenshot of {url} to {output_file_path}")
        
    try:
        page = await web_player["browser"].new_page()
        await page.goto(url)
        await page.wait_for_timeout(load_time)
        await page.screenshot(path=output_file_path, full_page=True, animations='disabled')
        await page.close()
    except Exception as e:
        print(f"Failed to take screenshot due to: {e}. Generating a blank image.")
        # Generate a blank image 
        img = Image.new('RGB', (1280, 960), color = 'white')
        img.save(output_file_path)

