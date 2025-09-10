"""
Training loop for RL recommender agent.
"""

from .utils import logger
from .config import EPISODES, TARGET_UPDATE_FREQ


class RLTrainer:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def train(self, episodes=EPISODES):
        rewards_per_episode = []

        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                self.agent.store_transition(state, action, reward, next_state, done)
                loss = self.agent.update()

                state = next_state
                total_reward += reward

            self.agent.update_epsilon()

            if ep % TARGET_UPDATE_FREQ == 0:
                self.agent.update_target_network()

            rewards_per_episode.append(total_reward)
            logger.info(f"Episode {ep+1}/{episodes} - Reward: {total_reward}, Epsilon: {self.agent.epsilon:.3f}")

        return rewards_per_episode
