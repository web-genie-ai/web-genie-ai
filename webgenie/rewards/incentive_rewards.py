import bittensor as bt
import numpy as np

def get_incentive_rewards(scores: np.ndarray, base_reward=100, alpha=1.5) -> np.ndarray:
    """
    Calculate rewards based on the piecewise linear with exponential growth method,
    preserving the original order of the scores, and returning rewards as a NumPy array.

    Parameters:
    - scores: NumPy array of raw scores.
    - base_reward: The minimum reward for the highest rank (rank 1).
    - alpha: The exponential scaling factor for ranks above the threshold.
    
    Returns:
    - rewards: NumPy array of rewards corresponding to the original order of scores.
    """
    bt.logging.debug(f"Scores: {scores}")
    threshold = scores.shape[0] // 2

    # Ensure input is a NumPy array
    scores = np.array(scores)
    
    # Rank players based on scores (highest score gets better rank)
    sorted_scores = np.sort(scores)  # Sort in ascending order
    score_to_rank = {score: idx + 1 for idx, score in enumerate(sorted_scores)}  # Map each score to its rank
    # Calculate rewards for each score based on its rank
    rewards = np.zeros_like(scores, dtype=float)  # Initialize the rewards array
    
    for idx, score in enumerate(scores):
        rank = score_to_rank[score]
        
        if rank <= threshold:
            # Linear reward scaling for ranks <= threshold
            reward = base_reward + (rank - 1) * (base_reward / 2)  # Linear scaling
        else:
            # Exponential reward scaling for ranks > threshold
            reward = base_reward * (alpha ** (rank - threshold))  # Exponential scaling
        
        rewards[idx] = reward  # Assign reward to the corresponding index

    bt.logging.debug(f"Rewards: {rewards}")
    return rewards


if __name__ == "__main__":
    scores = np.array([500, 450, 400, 750, 300, 250, 200])  # Raw scores as a NumPy array
    rewards = get_incentive_rewards(scores, base_reward=100, alpha=1.5)

    for score, reward in zip(scores, rewards):
        print(f"Score: {score}, Reward = {reward:.2f}")