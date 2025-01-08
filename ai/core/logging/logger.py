import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class AILogger:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_dir = Path(config.get('log_dir', 'logs'))
        self.app_name = config.get('app_name', 'ai_system')
        self.max_file_size = config.get('max_file_size', 10 * 1024 * 1024)  # 10MB
        self.backup_count = config.get('backup_count', 5)
        self.log_level = config.get('log_level', logging.INFO)

        self._setup_logging()

    def _setup_logging(self) -> None:
        """راه‌اندازی سیستم لاگینگ"""

        # ایجاد دایرکتوری لاگ‌ها
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # تنظیم لاگر اصلی
        logger = logging.getLogger(self.app_name)
        logger.setLevel(self.log_level)
        logger.propagate = False

        # فرمت لاگ‌ها
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # هندلر فایل چرخشی بر اساس سایز
        file_handler = RotatingFileHandler(
            self.log_dir / f'{self.app_name}.log',
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # هندلر فایل چرخشی روزانه برای لاگ‌های خطا
        error_handler = TimedRotatingFileHandler(
            self.log_dir / f'{self.app_name}_error.log',
            when='midnight',
            interval=1,
            backupCount=30
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

        # هندلر کنسول
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        self.logger = logger

    def _format_log_message(self, message: str, extra: Optional[Dict] = None) -> str:
        """قالب‌بندی پیام لاگ با اطلاعات اضافی"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'message': message
        }
        if extra:
            log_data.update(extra)
        return json.dumps(log_data, ensure_ascii=False)

    def info(self, message: str, extra: Optional[Dict] = None) -> None:
        """ثبت لاگ با سطح INFO"""
        self.logger.info(self._format_log_message(message, extra))

    def error(self, message: str, error: Optional[Exception] = None,
              extra: Optional[Dict] = None) -> None:
        """ثبت لاگ با سطح ERROR"""
        log_data = extra or {}
        if error:
            log_data.update({
                'error_type': error.__class__.__name__,
                'error_message': str(error)
            })
        self.logger.error(self._format_log_message(message, log_data), exc_info=error)

    def warning(self, message: str, extra: Optional[Dict] = None) -> None:
        """ثبت لاگ با سطح WARNING"""
        self.logger.warning(self._format_log_message(message, extra))

    def debug(self, message: str, extra: Optional[Dict] = None) -> None:
        """ثبت لاگ با سطح DEBUG"""
        self.logger.debug(self._format_log_message(message, extra))

    def critical(self, message: str, error: Optional[Exception] = None,
                 extra: Optional[Dict] = None) -> None:
        """ثبت لاگ با سطح CRITICAL"""
        log_data = extra or {}
        if error:
            log_data.update({
                'error_type': error.__class__.__name__,
                'error_message': str(error)
            })
        self.logger.critical(self._format_log_message(message, log_data), exc_info=error)

    def get_logs(self, level: Optional[str] = None,
                 limit: Optional[int] = None) -> list:
        """خواندن لاگ‌های ثبت شده"""
        logs = []
        log_file = self.log_dir / f'{self.app_name}.log'

        if not log_file.exists():
            return logs

        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                    if level and log_entry.get('level') != level:
                        continue
                    logs.append(log_entry)
                except json.JSONDecodeError:
                    continue

        if limit:
            logs = logs[-limit:]

        return logs

    def cleanup_old_logs(self, days: int = 30) -> None:
        """پاکسازی لاگ‌های قدیمی"""
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)

        for log_file in self.log_dir.glob(f'{self.app_name}*.log.*'):
            if log_file.stat().st_mtime < cutoff_date:
                try:
                    log_file.unlink()
                except Exception as e:
                    self.error(f"Failed to delete old log file {log_file}: {e}")