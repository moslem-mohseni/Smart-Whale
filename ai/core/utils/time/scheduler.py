import time
import threading
import logging
from typing import Callable

class TaskScheduler:
    def __init__(self):
        """
        مدیریت زمان‌بندی اجرای وظایف
        """
        self.logger = logging.getLogger("TaskScheduler")

    def run_at_interval(self, function: Callable, interval: int):
        """ اجرای یک وظیفه در بازه‌های زمانی مشخص (Interval Scheduling) """
        def loop():
            while True:
                try:
                    self.logger.info(f"✅ اجرای وظیفه: {function.__name__}")
                    function()
                except Exception as e:
                    self.logger.error(f"❌ خطا در اجرای وظیفه {function.__name__}: {e}")
                time.sleep(interval)

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
        self.logger.info(f"🔄 وظیفه {function.__name__} در بازه {interval} ثانیه‌ای برنامه‌ریزی شد.")

    def run_at_time(self, function: Callable, run_time: int):
        """ اجرای یک وظیفه در یک زمان مشخص (One-Time Scheduling) """
        def delay_and_run():
            time.sleep(run_time - time.time())
            try:
                self.logger.info(f"✅ اجرای وظیفه: {function.__name__}")
                function()
            except Exception as e:
                self.logger.error(f"❌ خطا در اجرای وظیفه {function.__name__}: {e}")

        thread = threading.Thread(target=delay_and_run, daemon=True)
        thread.start()
        self.logger.info(f"⏳ وظیفه {function.__name__} در {run_time} اجرا خواهد شد.")
