import bittensor as bt
import asyncio
import os
import re
import uuid

from bs4 import BeautifulSoup
from lxml import etree
from lxml.etree import XMLSyntaxError
from PIL import Image
from playwright.async_api import async_playwright

from webgenie.constants import (
    WORK_DIR,
    CHROME_HTML_LOAD_TIME,
    PLACE_HOLDER_IMAGE_URL,
)
from webgenie.helpers.images import image_to_base64
    

def is_valid_resources(html_content: str) -> bool:
    """
    Check if the resources in the HTML content are valid.
    """
    # List of allowed patterns for CSS and JavaScript resources
    allowed_patterns = [
        r"https?://cdn.jsdelivr.net/npm/tailwindcss@[^/]+/dist/tailwind.min.css",
        r"https?://stackpath.bootstrapcdn.com/bootstrap/[^/]+/css/bootstrap.min.css",
        r"https?://code.jquery.com/jquery-[^/]+.min.js",
        r"https?://stackpath.bootstrapcdn.com/bootstrap/[^/]+/js/bootstrap.bundle.min.js",
    ]
    
    soup = BeautifulSoup(html_content, 'html.parser')
    resources = soup.find_all(['link', 'script'])
    
    for resource in resources:
        if resource.name == 'link' and resource.get('rel') == ['stylesheet']:
            href = resource.get('href')
            if href and not any(re.match(pattern, href) for pattern in allowed_patterns):
                return False
        elif resource.name == 'script':
            src = resource.get('src')
            if src and not any(re.match(pattern, src) for pattern in allowed_patterns):
                return False

    return True


def is_valid_html(html_content: str) -> bool:
    """
    Check if the HTML is valid.
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        return True
    except Exception as e:
        bt.logging.error(f"An error occurred: {e}")
        return False


def seperate_html_css(html_content: str) -> tuple[str, str]: 
    """
    Seperate the HTML and CSS from the HTML content.
    """
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


async def html_to_screenshot(html_content: str, page_load_time: int = 1000) -> str:
    """
    Take a screenshot of the HTML content.
    """
    html_path = f"{WORK_DIR}/screenshot_{uuid.uuid4()}.html"
    with open(html_path, "w") as f:
        f.write(html_content)
    png_path = f"{WORK_DIR}/screenshot_{uuid.uuid4()}.png"
    url = f"file://{os.path.abspath(html_path)}"
    
    try:
        async with async_playwright() as p:
            # Choose a browser, e.g., Chromium, Firefox, or WebKit
            browser = await p.chromium.launch()
            page = await browser.new_page()

            # Navigate to the URL
            await page.goto(url, timeout=CHROME_HTML_LOAD_TIME)
            await page.wait_for_timeout(page_load_time)
            # Take the screenshot
            await page.screenshot(
                path=png_path, 
                full_page=True, 
                animations="disabled", 
                timeout=CHROME_HTML_LOAD_TIME,
            )
            await page.close()
            await browser.close()
    except Exception as e: 
        print(f"Failed to take screenshot due to: {e}. Generating a blank image.")
        # Generate a blank image 
        img = Image.new('RGB', (1280, 960), color = 'white')
        img.save(png_path)

    await asyncio.sleep(0.1)
    base64_image = image_to_base64(png_path)
    
    await asyncio.sleep(0.1)
    os.remove(html_path)
    os.remove(png_path)
    return base64_image


def format_html(html_content: str) -> str:
    """
    Format the HTML content.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.prettify()


def replace_image_sources(html_content: str, new_url: str = PLACE_HOLDER_IMAGE_URL) -> str:
    """
    Replace the image sources in the HTML content.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Replace 'src' attribute in <img> tags
    for img_tag in soup.find_all('img'):
        img_tag['src'] = new_url
    
    # Replace 'srcset' attribute in <source> tags
    for source_tag in soup.find_all('source'):
        if 'srcset' in source_tag.attrs:
            source_tag['srcset'] = new_url
    
    # Replace URLs in inline styles (background-image) in elements
    for tag in soup.find_all(style=True):
        style = tag['style']
        # Match both background-image and shorthand background property
        updated_style = re.sub(r'background\s*:\s*[^;]*url\([^)]+\)', f'background: url({new_url})', style)
        updated_style = re.sub(r'background-image\s*:\s*url\([^)]+\)', f'background-image: url({new_url})', updated_style)
        tag['style'] = updated_style
    
    # Replace URLs in <style> blocks
    for style_tag in soup.find_all('style'):
        style_content = style_tag.string
        if style_content:
            # Update both shorthand background and background-image URLs
            updated_content = re.sub(r'background\s*:\s*[^;]*url\([^)]+\)', f'background: url({new_url})', style_content)
            updated_content = re.sub(r'background-image\s*:\s*url\([^)]+\)', f'background-image: url({new_url})', updated_content)
            style_tag.string.replace_with(updated_content)
    
    return str(soup)


def preprocess_html(html_content: str) -> str:
    """
    Preprocess the HTML content.
    """
    if not is_valid_html(html_content):
        return ""
    html_content = format_html(html_content)
    html_content = replace_image_sources(html_content)
    return html_content


def is_empty_html(html_content: str) -> bool:
    """Check if HTML body is empty or missing.
    
    Args:
        html (str): HTML string to check
        
    Returns:
        bool: True if body is empty or missing, False otherwise
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    body = soup.find('body')
    
    if not body:
        return True
        
    if not body.get_text(strip=True):
        return True
        
    return False
