import cv2
import numpy as np
from skimage import io, color
from skimage.feature import SIFT
from skimage.metrics import structural_similarity as ssim
from difflib import SequenceMatcher

from webgenie.rewards.visual_reward.common.color_diff import color_similarity_ciede2000
from webgenie.rewards.visual_reward.common.extract_html_elements import HTMLElement
from webgenie.rewards.visual_reward.common.sift import match_sift_features
from webgenie.rewards.visual_reward.common.color_diff import color_similarity_ciede2000
# similarity is 1 if they are the same, 0 if they are completely different

def calculate_color_similarity(
        original_element: HTMLElement, 
        predicted_element: HTMLElement,
    ):
    return color_similarity_ciede2000(original_element.color, predicted_element.color)

def calculate_text_similarity(
        original_element: HTMLElement, 
        predicted_element: HTMLElement,
    ):
    if not original_element.text and not predicted_element.text:
        return 1
    
    if not original_element.text or not predicted_element.text:
        return 0
    
    return SequenceMatcher(None, original_element.text, predicted_element.text).ratio()

def calculate_block_similarity(
        original_element: HTMLElement, 
        predicted_element: HTMLElement,
    ):

    x_shift = abs(predicted_element.scaled_bounding_box["x"] - original_element.scaled_bounding_box["x"])
    y_shift = abs(predicted_element.scaled_bounding_box["y"] - original_element.scaled_bounding_box["y"])
    xx_shift = abs(predicted_element.scaled_bounding_box["x"] + predicted_element.scaled_bounding_box["width"] 
                   - original_element.scaled_bounding_box["x"] - original_element.scaled_bounding_box["width"])
    yy_shift = abs(predicted_element.scaled_bounding_box["y"] + predicted_element.scaled_bounding_box["height"] 
                   - original_element.scaled_bounding_box["y"] - original_element.scaled_bounding_box["height"])

    return 1 - (x_shift + y_shift + xx_shift + yy_shift) / 4

def calculate_visual_similarity(
        predicted_element: HTMLElement, 
        original_element: HTMLElement,
    ):
    sift_score = match_sift_features(predicted_element.keypoints, predicted_element.descriptors, 
                                    original_element.keypoints, original_element.descriptors)
    avg_color_score = color_similarity_ciede2000(predicted_element.avg_color, original_element.avg_color)
    print(sift_score, avg_color_score)
    return sift_score * 0.5 + avg_color_score * 0.5
