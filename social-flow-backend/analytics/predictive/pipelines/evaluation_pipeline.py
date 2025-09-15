"""
Evaluation Pipeline
- Loads model artifact and test features, computes evaluation metrics and plots
- Persists evaluation report and artifacts
"""

import os
import pandas as pd
from .config import settings
from .utils import logger, ensure_dir, write_json
from analytics.predictive.models.evaluation import regression_metrics, plot_actual_vs_pred
from analytics.predictive.models.inference import predict_batch, load_model_artifact

class EvaluationPipeline:
    name = "evaluation"

    def __init__(self, model_name: str = "user_growth_xgb", test_suffix: str = "_test.parquet"):
        self.model_name = model_name
        self.test_path = os.path.join(settings.FEATURE_STORE, f"raw_events{test_suffix}")
        self.out_dir = os.path.join(settings.BASE_DIR, "runs", "evaluation")
        ensure_dir(self.out_dir)

    def dry_run(self):
        if not os.path.exists(self.test_path):
            logger.warning(f"Test features missing at {self.test_path}")

    def run(self):
        if not os.path.exists(self.test_path):
            raise FileNotFoundError(self.test_path)

        test_df = pd.read_parquet(self.test_path)
        y_col = test_df.columns[-1]
        X_test = test_df.drop(columns=[y_col])
        y_test = test_df[y_col]

        model_pipeline = load_model_artifact(self.model_name)
        preds = predict_batch(model_pipeline, X_test, batch_size=500)

        metrics = regression_metrics(y_test, preds)
        report_path = os.path.join(self.out_dir, f"{self.model_name}_eval.json")
        write_json(report_path, metrics)
        logger.info(f"Saved evaluation metrics to {report_path}")

        # plot actual vs pred
        plot_path = os.path.join(self.out_dir, f"{self.model_name}_actual_vs_pred.png")
        plot_actual_vs_pred(y_test, preds, out_path=plot_path)
        logger.info(f"Saved actual vs predicted plot to {plot_path}")

        return {"metrics": metrics, "plot": plot_path, "report": report_path}
