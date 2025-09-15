from .extract import Extractor
from .transform import Transformer
from .load import Loader
from .quality_checks import QualityChecker
from .monitoring import Monitor
from .utils import logger


class ETLRunner:
    """Main ETL Orchestration"""

    def __init__(self):
        self.extractor = Extractor()
        self.transformer = Transformer()
        self.loader = Loader()
        self.qc = QualityChecker()
        self.monitor = Monitor()

    def run(self):
        try:
            # 1. Extract
            user_records = self.extractor.from_postgres("SELECT id, username, email FROM users")
            video_records = self.extractor.from_kafka(max_messages=50)

            # 2. Transform
            user_records = self.transformer.clean_user_data(user_records)
            user_records = self.transformer.enrich_with_hash(user_records)
            video_records = self.transformer.transform_videos_spark(video_records)

            # 3. Quality checks
            self.qc.check_nulls(user_records, ["id", "username"])
            self.qc.check_duplicates(user_records, "id")
            self.qc.check_schema(video_records, ["id", "title"])

            # 4. Load
            self.loader.to_snowflake("users", user_records)
            self.loader.to_snowflake("videos", video_records)

            # 5. Monitoring
            self.monitor.push_metrics(len(user_records) + len(video_records))

            logger.info("ETL pipeline executed successfully ✅")
        except Exception as e:
            logger.exception(f"ETL pipeline failed: {e}")
            self.monitor.push_metrics(0, errors=1)


if __name__ == "__main__":
    ETLRunner().run()
