"""
RL-based recommender wrapper for inference.
"""

import torch
import numpy as np
from .utils import get_device


class RLRecommender:
    def __init__(self, agent, device="cuda"):
        self.agent = agent
        self.device = get_device(device)

    def recommend(self, state, top_k=5):
        """
        Recommend items given current state.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.agent.q_net(state_tensor).cpu().numpy().flatten()
        top_items = np.argsort(q_values)[::-1][:top_k]
        scores = q_values[top_items]
        return list(zip(top_items, scores))
