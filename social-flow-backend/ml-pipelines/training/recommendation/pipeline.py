# High-level orchestration pipeline
"""
pipeline.py
-----------
High-level orchestration of the recommendation training pipeline.
"""

from data_loader import DataLoader
from trainer import Trainer
from utils import set_seed, setup_logger


def main():
    set_seed(42)
    logger = setup_logger()

    logger.info("Loading data...")
    loader = DataLoader("data/interactions.csv", "data/items.csv")
    df = loader.load_data()
    train, val, test = loader.train_val_test_split(df)

    logger.info("Initializing trainer...")
    num_users, num_items = df["user_id"].nunique(), df["item_id"].nunique()
    trainer = Trainer(num_users=num_users, num_items=num_items, epochs=5)

    logger.info("Starting training...")
    trainer.fit(train, val)

    logger.info("Training completed.")


if __name__ == "__main__":
    main()
