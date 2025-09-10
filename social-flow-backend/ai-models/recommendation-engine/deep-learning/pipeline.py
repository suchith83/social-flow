"""
Pipeline for training and recommending with deep learning models.
"""

from .trainer import Trainer
from .recommender import DeepLearningRecommender
from .dataset import InteractionDataset


class DeepLearningPipeline:
    def __init__(self, model, loss_type="bce", device="cuda"):
        self.model = model
        self.trainer = Trainer(model, loss_type=loss_type, device=device)
        self.recommender = DeepLearningRecommender(model, device=device)

    def train(self, user_ids, item_ids, ratings, val_split=0.2):
        split_idx = int(len(user_ids) * (1 - val_split))
        train_dataset = InteractionDataset(user_ids[:split_idx], item_ids[:split_idx], ratings[:split_idx])
        val_dataset = InteractionDataset(user_ids[split_idx:], item_ids[split_idx:], ratings[split_idx:])

        self.trainer.train(train_dataset, val_dataset=val_dataset)

    def recommend(self, user_id, all_item_ids, known_items=set(), top_k=10):
        return self.recommender.recommend(user_id, all_item_ids, known_items, top_k)
