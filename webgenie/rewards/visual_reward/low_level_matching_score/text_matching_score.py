import numpy as np
from scipy.optimize import linear_sum_assignment

from webgenie.rewards.visual_reward.common.extract_html_elements import HTMLElement, extract_html_elements
from webgenie.rewards.visual_reward.common.similarity import (
    calculate_text_similarity,
    calculate_block_similarity,
    calculate_color_similarity,
)


def calculate_cost(predicted_element: HTMLElement, original_element: HTMLElement):
    text_similarity = calculate_text_similarity(predicted_element, original_element)
    block_similarity = calculate_block_similarity(predicted_element, original_element)
    return text_similarity * 0.8 + block_similarity * 0.2


def create_cost_matrix(predicted_elements, original_elements):
    n = len(predicted_elements)
    m = len(original_elements)
    cost_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cost_matrix[i][j] = -calculate_cost(predicted_elements[i], original_elements[j])
    return cost_matrix


def calculate_text_matching_similarity(predicted_elements, original_elements):
    cost_matrix = create_cost_matrix(predicted_elements, original_elements)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    match_count = 0
    text_similarity_sum = 0
    block_similarity_sum = 0
    color_similarity_sum = 0

    for i , j in zip(row_ind, col_ind):
        text_similarity = calculate_text_similarity(predicted_elements[i], original_elements[j])
        if text_similarity < 0.5:
            continue
        
        block_similarity = calculate_block_similarity(predicted_elements[i], original_elements[j])
        color_similarity = calculate_color_similarity(predicted_elements[i], original_elements[j])
        
        match_count += 1
        text_similarity_sum += text_similarity
        block_similarity_sum += block_similarity
        color_similarity_sum += color_similarity

    total_count = len(predicted_elements) + len(original_elements) - match_count
    if total_count == 0:
        return 1
    
    text_similarity_score = text_similarity_sum / total_count
    block_similarity_score = block_similarity_sum / total_count
    color_similarity_score = color_similarity_sum / total_count
    
    return text_similarity_score * 0.5 + block_similarity_score * 0.3 + color_similarity_score * 0.2