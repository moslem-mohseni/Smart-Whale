import time
import datetime

class TimeUtils:
    @staticmethod
    def get_current_utc() -> str:
        """ دریافت زمان فعلی به‌صورت UTC در فرمت استاندارد ISO 8601 """
        return datetime.datetime.utcnow().isoformat()

    @staticmethod
    def get_current_local() -> str:
        """ دریافت زمان فعلی به‌صورت Local Time در فرمت استاندارد """
        return datetime.datetime.now().isoformat()

    @staticmethod
    def timestamp_to_datetime(timestamp: int) -> str:
        """ تبدیل Timestamp به رشته‌ی Datetime استاندارد """
        return datetime.datetime.utcfromtimestamp(timestamp).isoformat()

    @staticmethod
    def datetime_to_timestamp(dt_str: str) -> int:
        """ تبدیل رشته‌ی Datetime استاندارد به Timestamp """
        return int(datetime.datetime.fromisoformat(dt_str).timestamp())

    @staticmethod
    def calculate_time_difference(time1: str, time2: str) -> str:
        """ محاسبه اختلاف بین دو زمان برحسب ثانیه """
        dt1 = datetime.datetime.fromisoformat(time1)
        dt2 = datetime.datetime.fromisoformat(time2)
        delta = abs((dt2 - dt1).total_seconds())
        return f"{delta} ثانیه"

    @staticmethod
    def wait(seconds: int):
        """ توقف اجرای برنامه به مدت مشخص (شبیه sleep) """
        time.sleep(seconds)
