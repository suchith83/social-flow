# Business rules & constraints
# constraints.py
import pandas as pd
from typing import Dict, Any, Callable
from .utils import logger, timed

class ConstraintChecker:
    """
    Enforces custom business rules on data.
    """

    def __init__(self, constraints: Dict[str, Callable[[pd.DataFrame], bool]]):
        """
        constraints: dictionary mapping rule_name -> validation function(df) -> bool
        """
        self.constraints = constraints

    @timed
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        results = {}
        for name, rule_fn in self.constraints.items():
            try:
                results[name] = rule_fn(df)
            except Exception as e:
                results[name] = f"Error: {str(e)}"
        return results
