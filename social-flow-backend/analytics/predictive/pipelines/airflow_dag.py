"""
Example Airflow DAG to run the predictive pipelines.
Drop this file into your Airflow DAGs directory (adjust imports and paths as needed).
This DAG is intentionally simple and demonstrates how to call the pipelines programmatically.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from .preprocessing_pipeline import PreprocessingPipeline
from .training_pipeline import TrainingPipeline
from .evaluation_pipeline import EvaluationPipeline
from .deployment_pipeline import DeploymentPipeline
from .monitoring_pipeline import MonitoringPipeline
from .orchestrator import Orchestrator
from .config import settings

default_args = {
    "owner": "analytics",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="predictive_model_full_cycle",
    default_args=default_args,
    schedule_interval=settings.SCHEDULE_INTERVAL,
    catchup=False,
    max_active_runs=1,
)

def run_preprocessing(**context):
    stage = PreprocessingPipeline(feature_table="raw_events", target_col="target")
    return stage.run()

def run_training(**context):
    stage = TrainingPipeline(feature_table="raw_events", model_name="user_growth_xgb")
    return stage.run()

def run_evaluation(**context):
    stage = EvaluationPipeline(model_name="user_growth_xgb")
    return stage.run()

def run_deployment(**context):
    stage = DeploymentPipeline(model_name="user_growth_xgb")
    return stage.run()

def run_monitoring(**context):
    # If you want to push metrics from evaluation output, you can parse XComs or load eval file
    monitor = MonitoringPipeline()
    return monitor.run()

preproc = PythonOperator(task_id="preprocessing", python_callable=run_preprocessing, dag=dag)
train = PythonOperator(task_id="training", python_callable=run_training, dag=dag)
evaluate = PythonOperator(task_id="evaluation", python_callable=run_evaluation, dag=dag)
deploy = PythonOperator(task_id="deployment", python_callable=run_deployment, dag=dag)
monitor = PythonOperator(task_id="monitoring", python_callable=run_monitoring, dag=dag)

preproc >> train >> evaluate >> deploy >> monitor
