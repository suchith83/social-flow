from .utils import logger


class QualityChecker:
    """Data quality validation checks"""

    def check_nulls(self, records: list[dict], fields: list[str]):
        """Ensure critical fields are not null"""
        for field in fields:
            nulls = sum(1 for r in records if r.get(field) is None)
            if nulls > 0:
                logger.warning(f"Found {nulls} NULLs in field {field}")
        logger.info("Null checks completed")

    def check_duplicates(self, records: list[dict], key_field: str):
        """Check for duplicate records"""
        seen = set()
        duplicates = 0
        for r in records:
            val = r.get(key_field)
            if val in seen:
                duplicates += 1
            else:
                seen.add(val)
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicates on {key_field}")
        logger.info("Duplicate checks completed")

    def check_schema(self, records: list[dict], expected_fields: list[str]):
        """Ensure schema consistency"""
        for r in records:
            missing = [f for f in expected_fields if f not in r]
            if missing:
                logger.error(f"Schema mismatch, missing fields: {missing}")
        logger.info("Schema checks completed")
