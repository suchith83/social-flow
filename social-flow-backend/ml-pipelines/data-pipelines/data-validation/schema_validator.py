# Enforces data schema consistency
# schema_validator.py
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
from typing import Dict, Any
from .utils import logger, timed

class SchemaValidator:
    """
    Validates dataframes against predefined schemas using Pandera.
    """

    def __init__(self, schema: pa.DataFrameSchema):
        self.schema = schema

    @timed
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            validated_df = self.schema.validate(df, lazy=True)
            return {"status": "success", "validated_rows": len(validated_df)}
        except pa.errors.SchemaErrors as e:
            logger.error(f"Schema validation failed: {e.failure_cases}")
            return {
                "status": "failed",
                "errors": e.failure_cases.to_dict("records")
            }
