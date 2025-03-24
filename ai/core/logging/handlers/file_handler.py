import logging
from logging.handlers import RotatingFileHandler

class FileLogHandler:
    def __init__(self, log_file="system.log", max_size=5 * 1024 * 1024, backup_count=3, log_level=logging.INFO):
        """
        مدیریت لاگ‌های سیستم و ذخیره در فایل با قابلیت چرخش خودکار
        :param log_file: نام فایل لاگ
        :param max_size: حداکثر اندازه فایل لاگ قبل از چرخش (پیش‌فرض 5MB)
        :param backup_count: تعداد فایل‌های لاگ قدیمی که ذخیره می‌شوند
        :param log_level: سطح لاگ (پیش‌فرض: INFO)
        """
        self.logger = logging.getLogger("FileLogger")
        self.logger.setLevel(log_level)

        # تنظیم چرخش لاگ‌ها
        file_handler = RotatingFileHandler(log_file, maxBytes=max_size, backupCount=backup_count)
        file_handler.setLevel(log_level)

        # تنظیم فرمت لاگ‌ها
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def log(self, level, message):
        """ ثبت لاگ در سطح مشخص شده """
        if level == "debug":
            self.logger.debug(message)
        elif level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "critical":
            self.logger.critical(message)

    def info(self, message):
        """ ثبت لاگ در سطح INFO """
        self.logger.info(message)

    def error(self, message):
        """ ثبت لاگ در سطح ERROR """
        self.logger.error(message)
