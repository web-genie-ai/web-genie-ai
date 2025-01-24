import bittensor as bt
import os
import asyncio
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from typing import Any
from skimage import io, color

from webgenie.constants import (
    DEFAULT_LOAD_TIME, 
    CHROME_HTML_LOAD_TIME,
    JAVASCRIPT_RUNNING_TIME,
)
from webgenie.rewards.visual_reward.common.browser import web_player
from webgenie.rewards.visual_reward.common.sift import extract_sift_from_roi


class HTMLElement(BaseModel):
    text: str = Field(default="")
    bounding_box: dict = Field(default={})
    scaled_bounding_box: dict = Field(default={})
    color: tuple[int, int, int] = Field(default=(0, 0, 0))
    input_type: str = Field(default="")
    input_placeholder: str = Field(default="")

    keypoints: Any = Field(default=None)
    descriptors: Any = Field(default=None)
    avg_color: tuple[int, int, int] = Field(default=(0, 0, 0))


def parse_rgb_string(rgb_str: str) -> tuple[int, int, int]:
    """Convert RGB/RGBA color string like 'rgb(23, 34, 45)' or 'rgba(23, 34, 45, 0.5)' to (23, 34, 45) tuple."""
    try:
        # Handle both rgb and rgba formats
        rgb_str = rgb_str.strip()
        if rgb_str.startswith('rgba'):
            rgb_str = rgb_str.removeprefix('rgba(')
        else:
            rgb_str = rgb_str.removeprefix('rgb(')
        rgb_str = rgb_str.removesuffix(')')
        
        # Split and take only the RGB values, ignoring alpha if present
        values = rgb_str.split(',')[:3]
        return tuple(int(v.strip()) for v in values)
    except Exception as e:
        bt.logging.error(f"Error parsing rgb string: {e}")
        return (0, 0, 0)


async def extract_html_elements(file_path, load_time = DEFAULT_LOAD_TIME):
    if os.path.exists(file_path):
        url = f"file:///{os.path.abspath(file_path)}"

    screenshot_path = file_path.replace(".html", ".png")

    text_elements = []
    button_elements = []
    input_elements = []
    anchor_elements = []        
    try:
        page = await web_player["browser"].new_page()

        await page.goto(url, timeout=CHROME_HTML_LOAD_TIME)

        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(JAVASCRIPT_RUNNING_TIME)
        
        if not os.path.exists(screenshot_path):
            await page.screenshot(
                path=screenshot_path, 
                full_page=True, 
                animations="disabled", 
                timeout=CHROME_HTML_LOAD_TIME,
            )
            
        with open(screenshot_path, "rb") as f:
            screenshot = Image.open(f)
            W, H = screenshot.size

        async def add_element(node, has_children):
            # Combine all necessary evaluations into one to reduce overhead
            rendered_data = await node.evaluate("""
                (el) => {
                    const styles = window.getComputedStyle(el);
                    const type = el.getAttribute('type') || 'text';
                    const placeholder = el.getAttribute('placeholder') || '';
                    return {
                        tagName: el.tagName.toLowerCase(),
                        color: styles.color || 'rgb(0, 0, 0)',
                        type: type,
                        placeholder: placeholder
                    };
                }
            """)
            
            # Extract all relevant data from the evaluated result
            text = await node.inner_text()
            bounding_box = await node.bounding_box()
            
            # Early return if no bounding box or invalid dimensions
            if bounding_box is None or bounding_box["width"] <= 0 or bounding_box["height"] <= 0:
                return

            scaled_bounding_box = {
                "x": bounding_box["x"] / W,
                "y": bounding_box["y"] / H,
                "width": bounding_box["width"] / W,
                "height": bounding_box["height"] / H
            }

            # Create the HTMLElement object with the extracted data
            element_data = HTMLElement(
                text=text, 
                bounding_box=bounding_box, 
                scaled_bounding_box=scaled_bounding_box,
            )

            # Add the element based on its tag name
            if rendered_data['tagName'] == "button":
                button_elements.append(element_data)
            elif rendered_data['tagName'] == "input":
                # Additional input-specific properties
                element_data.input_type = rendered_data['type']
                element_data.input_placeholder = rendered_data['placeholder']
                input_elements.append(element_data)
            elif rendered_data['tagName'] == "a":
                anchor_elements.append(element_data)

            # Add to text elements only if no children
            if not has_children:
                text_elements.append(
                    HTMLElement(
                        text=text, 
                        bounding_box=bounding_box, 
                        scaled_bounding_box=scaled_bounding_box,
                        color=parse_rgb_string(rendered_data['color']),
                    )
                )
                    
        async def traverse(node):
            stack = [node]
            while stack:
                current_node = stack.pop()
                children = await current_node.query_selector_all(':scope > *')
                for child in children:
                    stack.append(child)
                try:
                    await add_element(current_node, bool(children))
                except Exception as e:
                    #bt.logging.warning(f"Error adding element: {e}")
                    pass
                # Dispose the node when done
                await current_node.dispose()
            
        await traverse(await page.query_selector('body'))
        await page.close()
        preprocess_html_elements(file_path, button_elements)
        preprocess_html_elements(file_path, input_elements)
        preprocess_html_elements(file_path, anchor_elements)    
    except Exception as e:
        bt.logging.error(f"Error extracting html elements from {file_path}: {e}")
    return text_elements, button_elements, input_elements, anchor_elements


def preprocess_html_elements(html_path, html_elements):
    image_path = html_path.replace(".html", ".png")
    color_image = io.imread(image_path)
    for element in html_elements:
        bbox = element.bounding_box
        x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["width"]), int(bbox["height"]) 
        try:
            element.avg_color = np.mean(color_image[y:y+h, x:x+w], axis=(0, 1))
        except Exception as e:
            #bt.logging.warning(f"Error calculating avg color of html elements from {html_path}: {e}")
            element.avg_color = (0, 0, 0)

    gray_image = color.rgb2gray(color_image)
    for element in html_elements:
        bbox = element.bounding_box
        x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["width"]), int(bbox["height"])
        try:
            element.keypoints, element.descriptors = extract_sift_from_roi(gray_image, (x, y, w, h))   
        except Exception as e:
            #bt.logging.warning(f"Error extracting sift from html elements from {html_path}: {e}")
            element.keypoints = None
            element.descriptors = None
