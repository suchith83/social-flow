# time_utils.py
import datetime
import pytz


class TimeUtils:
    """
    Utilities for time and date handling with timezone support.
    """

    @staticmethod
    def now_utc() -> datetime.datetime:
        return datetime.datetime.utcnow().replace(tzinfo=pytz.UTC)

    @staticmethod
    def now_local(tz="Asia/Kolkata") -> datetime.datetime:
        return datetime.datetime.now(pytz.timezone(tz))

    @staticmethod
    def format(dt: datetime.datetime, fmt="%Y-%m-%d %H:%M:%S") -> str:
        return dt.strftime(fmt)

    @staticmethod
    def parse(date_str: str, fmt="%Y-%m-%d %H:%M:%S") -> datetime.datetime:
        return datetime.datetime.strptime(date_str, fmt)

    @staticmethod
    def diff_seconds(dt1: datetime.datetime, dt2: datetime.datetime) -> int:
        return int((dt2 - dt1).total_seconds())
