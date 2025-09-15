"""
Monitoring Pipeline
- Emits simple health checks and metrics for model performance (offline)
- Pushes to Prometheus Pushgateway if configured
"""

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from .config import settings
from .utils import logger

class MonitoringPipeline:
    name = "monitoring"

    def __init__(self):
        self.registry = CollectorRegistry()
        self.eval_mae = Gauge("model_eval_mae", "Model evaluation MAE", registry=self.registry)
        self.eval_rmse = Gauge("model_eval_rmse", "Model evaluation RMSE", registry=self.registry)
        self.active_models = Gauge("deployed_models", "Number of deployed models", registry=self.registry)

    def dry_run(self):
        logger.info("Monitoring dry-run: checking pushgateway availability")
        if not settings.PROMETHEUS_PUSHGATEWAY:
            logger.warning("PROMETHEUS_PUSHGATEWAY not configured")

    def push_metrics(self, mae: float, rmse: float, deployed_count: int = 1):
        if not settings.PROMETHEUS_PUSHGATEWAY:
            logger.warning("No pushgateway configured; skipping metrics push")
            return
        self.eval_mae.set(mae)
        self.eval_rmse.set(rmse)
        self.active_models.set(deployed_count)
        push_to_gateway(settings.PROMETHEUS_PUSHGATEWAY, job="predictive_pipelines", registry=self.registry)
        logger.info(f"Pushed monitoring metrics MAE={mae} RMSE={rmse} deployed={deployed_count}")

    def run(self):
        # no-op runner - typically used by orchestration after evaluation
        logger.info("Monitoring pipeline run invoked - no-op by default")
        return True
