import bittensor as bt
import os
from bs4 import BeautifulSoup
from lxml import etree
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

def is_valid_html(html: str):
    try:
        soup = BeautifulSoup(html, 'html.parser')
        return True
    except Exception as e:
        bt.logging.debug(f"Error during HTML parsing: {e}")
        return False

def seperate_html_css(html_content: str): 
    soup = BeautifulSoup(html_content, 'html.parser')

    css = ''
    for style_tag in soup.find_all('style'):
        css += style_tag.get_text()
    for style_tag in soup.find_all('style'):
        style_tag.decompose()

    head = soup.head
    if not head:
        head = soup.new_tag('head')
        soup.html.insert(0, head)

    link_tag = soup.new_tag('link', rel='stylesheet', href='styles.css')
    head.append(link_tag)
    cleaned_html = str(soup)
    return cleaned_html, css

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
    return soup.prettify()

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
    if not is_valid_html(html):
        return ""
    html = beautify_html(html)
    html = replace_image_sources(html)
    return html

def is_empty_html(html: str) -> bool:
    """Check if HTML body is empty or missing.
    
    Args:
        html (str): HTML string to check
        
    Returns:
        bool: True if body is empty or missing, False otherwise
    """
    soup = BeautifulSoup(html, 'html.parser')
    body = soup.find('body')
    
    # Return True if no body tag exists
    if not body:
        return True
        
    # Return True if body has no content (whitespace is stripped)
    if not body.get_text(strip=True):
        return True
        
    return False

if __name__ == "__main__":
    html = """
    <html>
        <body>
            <h1>Hello, World!</h1>
        </body>
    </html>
    """

    print(preprocess_html(html))