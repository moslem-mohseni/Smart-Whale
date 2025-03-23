# infrastructure/clickhouse/config/settings.py

from dataclasses import dataclass
from typing import Optional, Dict, List, Any


@dataclass
class ClickHouseConfig:
    """
    تنظیمات اتصال و پیکربندی ClickHouse

    این کلاس تمام پارامترهای مورد نیاز برای برقراری اتصال به ClickHouse
    و تنظیم رفتار آن را نگهداری می‌کند.
    """
    # تنظیمات پایه اتصال
    host: str
    port: int
    database: str
    user: str
    password: str

    # تنظیمات پیشرفته
    secure: bool = False  # استفاده از SSL/TLS
    verify: bool = True  # تأیید گواهی SSL
    compression: bool = True  # فعال‌سازی فشرده‌سازی
    compress_block_size: int = 1048576  # اندازه بلوک فشرده‌سازی

    # تنظیمات کارایی
    min_connections: int = 1
    max_connections: int = 10
    connect_timeout: float = 10.0
    read_timeout: float = 30.0

    # تنظیمات پیش‌فرض پرس‌وجو
    settings: Optional[Dict[str, Any]] = None

    def get_connection_params(self) -> dict:
        """
        تولید پارامترهای اتصال برای کتابخانه clickhouse-driver

        Returns:
            dict: پارامترهای اتصال به فرمت مناسب برای کتابخانه
        """
        params = {
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'user': self.user,
            'password': self.password,
            'secure': self.secure,
            'verify': self.verify,
            'compression': self.compression,
            'compress_block_size': self.compress_block_size,
            'connect_timeout': self.connect_timeout,
            'send_receive_timeout': self.read_timeout,
        }

        if self.settings:
            params['settings'] = self.settings

        return params

    def get_dsn(self) -> str:
        """
        تولید رشته اتصال (DSN)

        Returns:
            str: رشته اتصال به فرمت مناسب
        """
        protocol = 'https' if self.secure else 'http'
        return f"{protocol}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class QuerySettings:
    """
    تنظیمات پیش‌فرض برای پرس‌وجوها

    این کلاس تنظیمات پیش‌فرضی که باید روی پرس‌وجوها اعمال شود را
    نگهداری می‌کند. این تنظیمات می‌توانند برای بهینه‌سازی عملکرد استفاده شوند.
    """
    max_execution_time: int = 60  # حداکثر زمان اجرا (ثانیه)
    max_threads: int = 8  # حداکثر تعداد thread برای هر پرس‌وجو
    max_memory_usage: int = 10_000_000_000  # حداکثر مصرف حافظه (بایت)
    max_rows_to_read: Optional[int] = None  # محدودیت تعداد سطرهای خوانده شده

    def to_dict(self) -> Dict[str, Any]:
        """تبدیل تنظیمات به دیکشنری"""
        return {
            'max_execution_time': self.max_execution_time,
            'max_threads': self.max_threads,
            'max_memory_usage': self.max_memory_usage,
            'max_rows_to_read': self.max_rows_to_read
        }