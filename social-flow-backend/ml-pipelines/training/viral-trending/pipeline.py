# High-level orchestration
"""
pipeline.py
-----------
High-level orchestration of the viral-trending training pipeline.
"""

from data_loader import ViralDataLoader
from feature_engineering import FeatureEngineer
from trainer import Trainer
from utils import set_seed, setup_logger


def main():
    set_seed(42)
    logger = setup_logger()

    logger.info("Loading data...")
    loader = ViralDataLoader("data/posts.csv", "data/engagements.csv")
    df = loader.load()
    train, val, test = loader.split(df)

    logger.info("Extracting features...")
    fe = FeatureEngineer()
    X_train, y_train = fe.fit_transform(train)
    X_val, y_val = fe.transform(val), val["viral"].values
    X_test, y_test = fe.transform(test), test["viral"].values

    logger.info("Initializing trainer...")
    trainer = Trainer(input_dim=X_train.shape[1], epochs=5)

    logger.info("Starting training...")
    trainer.fit(X_train, y_train, X_val, y_val)

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
