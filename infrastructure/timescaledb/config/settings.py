# infrastructure/timescaledb/config/settings.py

from dataclasses import dataclass
from typing import Optional


@dataclass
class TimescaleDBConfig:
    """کلاس تنظیمات اتصال به TimescaleDB

    این کلاس تمام پارامترهای مورد نیاز برای برقراری اتصال به TimescaleDB را نگهداری می‌کند.
    با استفاده از dataclass، مدیریت و اعتبارسنجی این پارامترها ساده‌تر می‌شود.
    """
    host: str
    port: int
    database: str
    user: str
    password: str
    min_connections: int = 1
    max_connections: int = 10
    connection_timeout: int = 30

    def get_connection_string(self) -> str:
        """ساخت رشته اتصال به پایگاه داده

        Returns:
            str: رشته اتصال به فرمت مناسب برای asyncpg
        """
        return (f"postgresql://{self.user}:{self.password}@{self.host}:"
                f"{self.port}/{self.database}")