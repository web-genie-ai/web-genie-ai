import os
import time
import uuid
from webgenie.constants import SCREENSHOT_SCRIPT_PATH, WORK_DIR
from webgenie.helpers.images import image_to_base64

def html_to_screenshot(html: str) -> str:
    html_path = f"{WORK_DIR}/screenshot_{uuid.uuid4()}.html"
    with open(html_path, "w") as f:
        f.write(html)
    png_path = f"{WORK_DIR}/screenshot_{uuid.uuid4()}.png"
    os.system(f"python3 {SCREENSHOT_SCRIPT_PATH} --html {html_path} --png {png_path}")
    time.sleep(0.1)
    return image_to_base64(png_path)