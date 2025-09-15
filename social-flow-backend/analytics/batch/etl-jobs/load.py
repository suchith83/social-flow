import snowflake.connector
from .config import settings
from .utils import logger, retry


class Loader:
    """Handles loading data into warehouse"""

    @retry(Exception, tries=3)
    def to_snowflake(self, table: str, records: list[dict]):
        """Load data into Snowflake"""
        if not records:
            logger.warning("No records to load into Snowflake")
            return

        conn = snowflake.connector.connect(settings.SNOWFLAKE_URI)
        cur = conn.cursor()

        # Create table dynamically if not exists
        columns = ", ".join(records[0].keys())
        placeholders = ", ".join(["%s"] * len(records[0]))
        create_stmt = f"CREATE TABLE IF NOT EXISTS {table} ({columns} STRING)"
        cur.execute(create_stmt)

        insert_stmt = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        for record in records:
            cur.execute(insert_stmt, list(record.values()))

        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Loaded {len(records)} records into {table}")
