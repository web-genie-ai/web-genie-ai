import os
import asyncio
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from typing import Any
from skimage import io, color

from webgenie.constants import DEFAULT_LOAD_TIME
from webgenie.rewards.visual_reward.common.browser import web_player
from webgenie.rewards.visual_reward.common.sift import extract_sift_from_roi


class HTMLElement(BaseModel):
    text: str = Field(default="")
    bounding_box: dict = Field(default={})
    scaled_bounding_box: dict = Field(default={})
    color: tuple[int, int, int] = Field(default=(0, 0, 0))
    input_type: str = Field(default="")
    input_placeholder: str = Field(default="")
    rendered_style: dict = Field(default={})

    keypoints: Any = Field(default=None)
    descriptors: Any = Field(default=None)
    avg_color: tuple[int, int, int] = Field(default=(0, 0, 0))


def parse_rgb_string(rgb_str: str) -> tuple[int, int, int]:
    """Convert RGB color string like 'rgb(23, 34, 45)' to (23, 34, 45) tuple."""
    # Extract numbers from rgb(r,g,b) format using string manipulation
    rgb_values = rgb_str.strip().removeprefix('rgb(').removesuffix(')').split(',')
    return tuple(int(v.strip()) for v in rgb_values)


async def extract_html_elements(file_path, load_time = DEFAULT_LOAD_TIME):
    if os.path.exists(file_path):
        url = f"file:///{os.path.abspath(file_path)}"
    text_elements = []
    button_elements = []
    input_elements = []
    anchor_elements = []
    try:
        page = await web_player["browser"].new_page()
    
        await page.goto(url)
        await page.wait_for_timeout(load_time)
        screenshot_path = file_path.replace(".html", ".png")
        await page.screenshot(path=screenshot_path, full_page=True, animations="disabled")
        await asyncio.sleep(10)
        with open(screenshot_path, "rb") as f:
            screenshot = Image.open(f)
            W, H = screenshot.size
        
        async def add_element(node, has_children):
            text = await node.inner_text()
            bounding_box = await node.bounding_box()
            rendered_style = await node.evaluate(
                """(el) => {
                    const styles = window.getComputedStyle(el);
                    return styles;
                }"""
            )
            tag_name = await node.evaluate("(node) => node.tagName.toLowerCase()")
            
            if bounding_box is None:
                return
            if bounding_box["width"] <= 0 or bounding_box["height"] <= 0:
                return

            scaled_bounding_box = {
                "x": bounding_box["x"] / W,
                "y": bounding_box["y"] / H,
                "width": bounding_box["width"] / W,
                "height": bounding_box["height"] / H
            }
            if tag_name == "button":
                button_elements.append(
                    HTMLElement(
                        text=text, 
                        bounding_box=bounding_box, 
                        scaled_bounding_box=scaled_bounding_box,
                        rendered_style=rendered_style,
                    )
                )
            elif tag_name == "input":
                input_type = await node.evaluate("(node) => node.getAttribute('type') || 'text'")
                input_placeholder = await node.evaluate("(node) => node.getAttribute('placeholder') || ''")
                input_elements.append(
                    HTMLElement(
                        text=text, 
                        bounding_box=bounding_box,
                        scaled_bounding_box=scaled_bounding_box,
                        rendered_style=rendered_style,
                        input_type=input_type,
                        input_placeholder=input_placeholder,
                    )
                )
            elif tag_name == "a":
                anchor_elements.append(
                    HTMLElement(
                        text=text, 
                        bounding_box=bounding_box, 
                        scaled_bounding_box=scaled_bounding_box,
                        rendered_style=rendered_style,
                    )
                )

            if not has_children:
                text_elements.append(
                    HTMLElement(
                        text=text, 
                        bounding_box=bounding_box, 
                        scaled_bounding_box=scaled_bounding_box,
                        color=parse_rgb_string(rendered_style["color"]),
                        rendered_style=rendered_style,
                    )
                )
        
        async def traverse(node):
            children = await node.query_selector_all(':scope > *')
            has_children = False
            for child in children:
                await traverse(child)
                has_children = True
            
            await add_element(node, has_children)
            
        await traverse(await page.query_selector('body'))
        await page.close()
    except Exception as e:
        print(e)
    preprocess_html_elements(file_path, button_elements)
    preprocess_html_elements(file_path, input_elements)
    preprocess_html_elements(file_path, anchor_elements)
    return text_elements, button_elements, input_elements, anchor_elements


def preprocess_html_elements(html_path, html_elements):
    image_path = html_path.replace(".html", ".png")
    color_image = io.imread(image_path)
    for element in html_elements:
        bbox = element.bounding_box
        x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["width"]), int(bbox["height"]) 
        element.avg_color = np.mean(color_image[y:y+h, x:x+w], axis=(0, 1))

    gray_image = color.rgb2gray(color_image)
    for element in html_elements:
        bbox = element.bounding_box
        x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["width"]), int(bbox["height"])
        element.keypoints, element.descriptors = extract_sift_from_roi(gray_image, (x, y, w, h))   
