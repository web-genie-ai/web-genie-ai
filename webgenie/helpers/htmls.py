import os
from bs4 import BeautifulSoup
import time
import re
import uuid
from webgenie.constants import (
    SCREENSHOT_SCRIPT_PATH,
    WORK_DIR,
    PLACE_HOLDER_IMAGE_URL,
    PYTHON_CMD
)
from webgenie.helpers.images import image_to_base64

def html_to_screenshot(html: str) -> str:
    html_path = f"{WORK_DIR}/screenshot_{uuid.uuid4()}.html"
    with open(html_path, "w") as f:
        f.write(html)
    png_path = f"{WORK_DIR}/screenshot_{uuid.uuid4()}.png"
    os.system(f"{PYTHON_CMD} {SCREENSHOT_SCRIPT_PATH} --html {html_path} --png {png_path}")
    
    time.sleep(0.1)
    base64_image = image_to_base64(png_path)

    time.sleep(0.1)
    os.remove(html_path)
    os.remove(png_path)
    return base64_image

def beautify_html(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    return str(soup)

def replace_image_sources(html_content, new_url = PLACE_HOLDER_IMAGE_URL):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for img_tag in soup.find_all('img'):
        img_tag['src'] = new_url
    
    for source_tag in soup.find_all('source'):
        if 'srcset' in source_tag.attrs:
            source_tag['srcset'] = new_url
    
    for tag in soup.find_all(style=True):
        style = tag['style']
        updated_style = re.sub(r'background-image\s*:\s*url\([^)]+\)', f'background-image: url({new_url})', style)
        tag['style'] = updated_style
    
    for style_tag in soup.find_all('style'):
        style_content = style_tag.string
        if style_content:
            updated_content = re.sub(r'background-image\s*:\s*url\([^)]+\)', f'background-image: url({new_url})', style_content)
            style_tag.string.replace_with(updated_content)
    
    return str(soup)

def preprocess_html(html: str) -> str:
    html = beautify_html(html)
    html = replace_image_sources(html)
    return html