import logging
from datetime import datetime


class TextLogFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[94m",  # آبی
        "INFO": "\033[92m",  # سبز
        "WARNING": "\033[93m",  # زرد
        "ERROR": "\033[91m",  # قرمز
        "CRITICAL": "\033[91m\033[1m",  # قرمز پررنگ
        "RESET": "\033[0m"  # ریست رنگ
    }

    def format(self, record):
        """
        فرمت کردن لاگ‌ها به‌صورت متنی خوانا برای Console و فایل
        :param record: رکورد لاگ
        :return: رشته فرمت‌شده
        """
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        log_level = record.levelname
        module = record.module
        message = record.getMessage()

        # رنگ‌بندی سطح لاگ‌ها
        color = self.COLORS.get(log_level, self.COLORS["RESET"])
        formatted_log = f"{timestamp} | {color}{log_level}{self.COLORS['RESET']} | {module} | {message}"

        return formatted_log
