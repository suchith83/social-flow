"""
Custom RL environment for recommendation.
"""

import numpy as np
from gym import Env, spaces


class RecommendationEnv(Env):
    """
    A simple recommendation environment:
    - State: user embedding + history of interactions
    - Action: choose an item to recommend
    - Reward: user click/engagement
    """

    def __init__(self, num_users, num_items, history_len=5):
        super(RecommendationEnv, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.history_len = history_len

        # Action space = item to recommend
        self.action_space = spaces.Discrete(num_items)

        # Observation space = user + history
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_users + history_len,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.current_user = np.random.randint(0, self.num_users)
        self.history = [0] * self.history_len
        return self._get_state()

    def step(self, action):
        # Simulated engagement probability
        click_prob = np.random.rand()
        reward = 1 if click_prob > 0.5 else 0  # binary click feedback

        # Update history
        self.history.pop(0)
        self.history.append(action)

        done = len(self.history) >= self.history_len
        return self._get_state(), reward, done, {}

    def _get_state(self):
        state = np.zeros(self.num_users + self.history_len)
        state[self.current_user] = 1  # one-hot encode user
        for i, item in enumerate(self.history):
            state[self.num_users + i] = item / self.num_items
        return state.astype(np.float32)
