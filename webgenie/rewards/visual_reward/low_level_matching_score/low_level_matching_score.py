import numpy as np

from webgenie.rewards.visual_reward.low_level_matching_score.element_matching_score import calculate_element_matching_similarity
from webgenie.rewards.visual_reward.low_level_matching_score.text_matching_score import calculate_text_matching_similarity
from webgenie.rewards.visual_reward.low_level_matching_score.input_matching_score import calculate_input_matching_similarity

from webgenie.rewards.visual_reward.common.extract_html_elements import extract_html_elements

async def low_level_matching_score(predict_html_path_list, original_html_path):
    
    (
        original_text_elements, 
        original_button_elements, 
        original_input_elements, 
        original_anchor_elements,
    ) = await extract_html_elements(original_html_path)

    results = []
    for predict_html_path in predict_html_path_list:
        (
            predicted_text_elements, 
            predicted_button_elements, 
            predicted_input_elements, 
            predicted_anchor_elements,
        ) = await extract_html_elements(predict_html_path)

        button_score = calculate_element_matching_similarity(predicted_button_elements, original_button_elements)
        anchor_score = calculate_element_matching_similarity(predicted_anchor_elements, original_anchor_elements)

        input_score = calculate_input_matching_similarity(predicted_input_elements, original_input_elements)
        text_score = calculate_text_matching_similarity(predicted_text_elements, original_text_elements)
        score = button_score * 0.25 + input_score * 0.25 + text_score * 0.25 + anchor_score * 0.25
        print(button_score, input_score, text_score, anchor_score)
        results.append(score)
    
    return np.array(results)
