import schedule
import time
from .report_generator import ReportGenerator
from .pdf_renderer import PDFRenderer
from .email_notifier import EmailNotifier
from .utils import logger


class ReportScheduler:
    """Schedule automated report generation and delivery"""

    def __init__(self):
        self.generator = ReportGenerator()
        self.renderer = PDFRenderer()
        self.notifier = EmailNotifier()

    def daily_user_engagement_report(self):
        path = self.generator.generate_report(
            template_name="user_engagement.html",
            query="SELECT user_id, COUNT(video_id) as videos_watched, SUM(duration) as total_watch_time FROM engagement GROUP BY user_id LIMIT 50",
            output_file="user_engagement.html",
        )
        pdf_path = self.renderer.render_pdf(path)
        self.notifier.send_report(
            to_email="analytics-team@socialflow.com",
            subject="Daily User Engagement Report",
            body="Attached is the latest user engagement report.",
            attachments=[pdf_path],
        )

    def weekly_video_performance_report(self):
        path = self.generator.generate_report(
            template_name="video_performance.html",
            query="SELECT video_id, COUNT(*) as views, AVG(watch_time) as avg_watch_time FROM video_stats GROUP BY video_id LIMIT 50",
            output_file="video_performance.html",
        )
        pdf_path = self.renderer.render_pdf(path)
        self.notifier.send_report(
            to_email="product-team@socialflow.com",
            subject="Weekly Video Performance Report",
            body="Attached is the latest weekly video performance report.",
            attachments=[pdf_path],
        )

    def start(self):
        """Schedule jobs"""
        schedule.every().day.at("06:00").do(self.daily_user_engagement_report)
        schedule.every().monday.at("07:00").do(self.weekly_video_performance_report)

        logger.info("Report scheduler started ✅")
        while True:
            schedule.run_pending()
            time.sleep(60)


if __name__ == "__main__":
    ReportScheduler().start()
