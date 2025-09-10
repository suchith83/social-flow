"""
Pipeline to train and evaluate RL recommender.
"""

from .trainer import RLTrainer
from .recommender import RLRecommender


class RLPipeline:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.trainer = RLTrainer(env, agent)
        self.recommender = RLRecommender(agent)

    def train(self, episodes=500):
        return self.trainer.train(episodes=episodes)

    def recommend(self, state, top_k=5):
        return self.recommender.recommend(state, top_k=top_k)
