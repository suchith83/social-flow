# Orchestrates validations as a pipeline
# pipeline_runner.py
import pandas as pd
from typing import Dict, Any
from .schema_validator import SchemaValidator
from .statistical_validator import StatisticalValidator
from .quality_checks import QualityChecker
from .constraints import ConstraintChecker
from .validation_report import ValidationReport
from .utils import logger, timed

class ValidationPipeline:
    """
    Orchestrates schema validation, statistical checks, quality checks, and constraints.
    """

    def __init__(
        self,
        schema_validator: SchemaValidator,
        statistical_validator: StatisticalValidator,
        quality_checker: QualityChecker,
        constraint_checker: ConstraintChecker,
    ):
        self.schema_validator = schema_validator
        self.statistical_validator = statistical_validator
        self.quality_checker = quality_checker
        self.constraint_checker = constraint_checker

    @timed
    def run(self, df: pd.DataFrame, reference_df: pd.DataFrame = None) -> Dict[str, Any]:
        results = {}

        logger.info("Running schema validation...")
        results["schema"] = self.schema_validator.validate(df)

        logger.info("Running quality checks...")
        results["quality"] = {
            "missing": self.quality_checker.check_missing(df),
            "duplicates": self.quality_checker.check_duplicates(df),
        }

        if reference_df is not None:
            logger.info("Running statistical drift detection...")
            results["statistics"] = {
                "drift": self.statistical_validator.detect_drift(df),
                "outliers": self.statistical_validator.detect_outliers(df),
            }

        logger.info("Running constraint checks...")
        results["constraints"] = self.constraint_checker.validate(df)

        return results

    def generate_report(self, results: Dict[str, Any], json_path: str, html_path: str):
        report = ValidationReport(results)
        report.to_json(json_path)
        report.to_html(html_path)
