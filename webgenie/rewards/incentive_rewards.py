import numpy as np

def get_incentive_rewards(scores: np.ndarray, base_reward=100, alpha=1.5) -> np.ndarray:
    threshold = scores.shape[0] // 2
    scores = np.array(scores)
    sorted_scores = np.sort(scores)  
    score_to_rank = {score: idx + 1 for idx, score in enumerate(sorted_scores)} 
    rewards = np.zeros_like(scores, dtype=float) 
    
    for idx, score in enumerate(scores):
        rank = score_to_rank[score]
        if rank <= threshold:
            reward = (rank - 1) * (base_reward / 2)  # Linear scaling
        else:
            reward = base_reward * (alpha ** (rank - threshold))  # Exponential scaling
        rewards[idx] = reward 

    return rewards


if __name__ == "__main__":
    scores = np.array([500, 450, 400, 750, 300, 250, 200])  # Raw scores as a NumPy array
    rewards = get_incentive_rewards(scores, base_reward=100, alpha=1.5)

    for score, reward in zip(scores, rewards):
        print(f"Score: {score}, Reward = {reward:.2f}")