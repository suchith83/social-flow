# Orchestrates ETL workflow
# pipeline.py
import pandas as pd
from typing import Dict, Any
from .extract import Extractor
from .transform import Transformer
from .load import Loader
from .utils import logger, timed, load_config

class ETLPipeline:
    """
    Orchestrates Extract -> Transform -> Load
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.extractor = Extractor(self.config)
        self.transformer = Transformer()
        self.loader = Loader(self.config)

    @timed
    def run(self) -> None:
        # --- Extract ---
        source = self.config["source"]
        if source["type"] == "sql":
            df = self.extractor.from_sql(source["query"], source["connection_uri"])
        elif source["type"] == "api":
            df = self.extractor.from_api(source["endpoint"], source.get("params", {}))
        elif source["type"] == "file":
            df = self.extractor.from_file(source["path"])
        else:
            raise ValueError("Unsupported source type")

        # --- Transform ---
        df = self.transformer.normalize_columns(df)
        df = self.transformer.clean_nulls(df, strategy=self.config["transform"].get("null_strategy", "drop"))
        df = self.transformer.apply_custom(df)

        # --- Load ---
        target = self.config["target"]
        if target["type"] == "sql":
            self.loader.to_sql(df, target["table"], target["connection_uri"])
        elif target["type"] == "s3":
            self.loader.to_s3(df, target["bucket"], target["key"], format=target.get("format", "parquet"))
        else:
            raise ValueError("Unsupported target type")
