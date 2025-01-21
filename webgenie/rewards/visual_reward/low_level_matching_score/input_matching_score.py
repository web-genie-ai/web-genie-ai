import bittensor as bt
import asyncio
import numpy as np
from math import sqrt
from difflib import SequenceMatcher
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity as ssim

from webgenie.rewards.visual_reward.common.extract_html_elements import HTMLElement
from webgenie.rewards.visual_reward.common.similarity import (
    calculate_block_similarity,
    calculate_visual_similarity,
    calculate_color_similarity
)


def calculate_cost(predicted_element: HTMLElement, original_element: HTMLElement):
    if predicted_element.input_type != original_element.input_type:
        return 0
    placeholder_similarity = SequenceMatcher(None, predicted_element.input_placeholder, original_element.input_placeholder).ratio()
    block_similarity = calculate_block_similarity(predicted_element, original_element)
    visual_similarity = calculate_visual_similarity(predicted_element, original_element)
    return placeholder_similarity * 0.5 + block_similarity * 0.3 + visual_similarity * 0.2


def create_cost_matrix(predicted_elements, original_elements):
    n = len(predicted_elements)
    m = len(original_elements)
    cost_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cost_matrix[i][j] = -calculate_cost(predicted_elements[i], original_elements[j])
    return cost_matrix


def calculate_input_matching_similarity(predicted_elements, original_elements):
    try:
        cost_matrix = create_cost_matrix(predicted_elements, original_elements)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        similarity_sum = 0
        for i , j in zip(row_ind, col_ind):
            similarity_sum += calculate_cost(predicted_elements[i], original_elements[j])
        
        total_count = max(len(predicted_elements), len(original_elements))
        if total_count == 0:
            return 1

        return similarity_sum / total_count
    except Exception as e:
        bt.logging.error(f"Error calculating input matching score: {e}")
        return 0

