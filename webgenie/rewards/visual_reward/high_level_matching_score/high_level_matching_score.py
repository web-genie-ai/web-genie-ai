import asyncio
import numpy as np

from webgenie.rewards.visual_reward.high_level_matching_score.clip_matching_score import calculate_clip_score
from webgenie.rewards.visual_reward.high_level_matching_score.histogram import histogram_matching_score


async def high_level_matching_score(predict_html_path_list, original_html_path):
    clip_score = await calculate_clip_score(predict_html_path_list, original_html_path)
    histogram_score = await histogram_matching_score(predict_html_path_list, original_html_path)

    return np.array(clip_score) * 0.5 + np.array(histogram_score) * 0.5

