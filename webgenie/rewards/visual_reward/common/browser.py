import bittensor as bt
from playwright.async_api import async_playwright


web_player = {
    "web_driver": None,
    "browser": None,
}


async def start_browser():
    global web_player
    web_driver = await async_playwright().start()
    browser = await web_driver.chromium.launch(headless=True)
    web_player["web_driver"] = web_driver
    web_player["browser"] = browser
    bt.logging.info(f"Started browser.")


async def stop_browser():
    global web_player
    await web_player["browser"].close()
    await web_player["web_driver"].stop()
    web_player["web_driver"] = None
    web_player["browser"] = None
    bt.logging.info(f"Stopped browser.")
