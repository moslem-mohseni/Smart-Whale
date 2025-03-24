# infrastructure/clickhouse/config/config.py
"""
ماژول مدیریت تنظیمات ClickHouse

این ماژول مسئول مدیریت متمرکز تمامی تنظیمات مورد نیاز برای ماژول ClickHouse است.
تمامی تنظیمات از متغیرهای محیطی یا فایل .env خوانده می‌شوند.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# بارگذاری متغیرهای محیطی از فایل .env
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class ClickHouseConfig:
    """
    کلاس مرکزی مدیریت تنظیمات ClickHouse

    این کلاس تمامی تنظیمات مربوط به اتصال، مکانیزم‌های خطایابی، امنیت و مدیریت داده را 
    در بر می‌گیرد.
    """
    # تنظیمات اتصال پایه
    host: str = os.getenv("CLICKHOUSE_HOST", "localhost")
    port: int = int(os.getenv("CLICKHOUSE_PORT", "9000"))
    database: str = os.getenv("CLICKHOUSE_DATABASE", "default")
    user: str = os.getenv("CLICKHOUSE_USER", "default")
    password: str = os.getenv("CLICKHOUSE_PASSWORD", "")
    secure: bool = os.getenv("CLICKHOUSE_SECURE", "False").lower() == "true"
    compression: bool = os.getenv("CLICKHOUSE_COMPRESSION", "True").lower() == "true"

    # تنظیمات پیشرفته اتصال
    max_connections: int = int(os.getenv("CLICKHOUSE_MAX_CONNECTIONS", "10"))
    connect_timeout: float = float(os.getenv("CLICKHOUSE_CONNECT_TIMEOUT", "10.0"))
    read_timeout: float = float(os.getenv("CLICKHOUSE_READ_TIMEOUT", "30.0"))

    # تنظیمات Circuit Breaker
    circuit_breaker_max_failures: int = int(os.getenv("CIRCUIT_BREAKER_MAX_FAILURES", "5"))
    circuit_breaker_reset_timeout: int = int(os.getenv("CIRCUIT_BREAKER_RESET_TIMEOUT", "60"))

    # تنظیمات Retry
    retry_max_attempts: int = int(os.getenv("RETRY_MAX_ATTEMPTS", "5"))
    retry_min_wait: int = int(os.getenv("RETRY_MIN_WAIT", "2"))
    retry_max_wait: int = int(os.getenv("RETRY_MAX_WAIT", "10"))

    # تنظیمات Load Balancer
    load_balancer_mode: str = os.getenv("LOAD_BALANCER_MODE", "random")  # random, round-robin, least-conn

    # تنظیمات امنیت
    access_control_secret: str = os.getenv("ACCESS_CONTROL_SECRET", "")
    access_token_expiry: int = int(os.getenv("ACCESS_TOKEN_EXPIRY", "3600"))
    encryption_key: str = os.getenv("ENCRYPTION_KEY", "")

    # تنظیمات مانیتورینگ
    prometheus_port: int = int(os.getenv("PROMETHEUS_PORT", "8000"))
    monitoring_interval: int = int(os.getenv("MONITORING_INTERVAL", "5"))

    # تنظیمات مدیریت داده
    data_retention_days: int = int(os.getenv("DATA_RETENTION_DAYS", "365"))
    backup_interval: int = int(os.getenv("BACKUP_INTERVAL", "86400"))  # 24 hours in seconds
    backup_dir: str = os.getenv("TIMESCALEDB_BACKUP_DIR", "backups")

    def __post_init__(self):
        """
        اعتبارسنجی تنظیمات بعد از ایجاد شی
        """
        self._validate_security_settings()
        self._validate_connection_settings()
        logger.info("ClickHouse configuration loaded successfully.")

    def _validate_security_settings(self):
        """
        بررسی اعتبار تنظیمات امنیتی
        """
        if not self.access_control_secret or len(self.access_control_secret) < 32:
            logger.warning("ACCESS_CONTROL_SECRET is not set or too short (less than 32 characters)")

        if not self.encryption_key or len(self.encryption_key) < 32:
            logger.warning("ENCRYPTION_KEY is not set or too short (less than 32 characters)")

    def _validate_connection_settings(self):
        """
        بررسی اعتبار تنظیمات اتصال
        """
        if self.load_balancer_mode not in ["random", "round-robin", "least-conn"]:
            logger.warning(f"Invalid LOAD_BALANCER_MODE: {self.load_balancer_mode}, falling back to 'random'")
            self.load_balancer_mode = "random"

    def get_connection_params(self) -> Dict[str, Any]:
        """
        تولید پارامترهای اتصال برای ClickHouse

        Returns:
            Dict[str, Any]: دیکشنری پارامترهای اتصال قابل استفاده در درایور ClickHouse
        """
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
            "secure": self.secure,
            "compression": self.compression,
            "connect_timeout": self.connect_timeout,
            "send_receive_timeout": self.read_timeout,
        }

    def get_dsn(self) -> str:
        """
        تولید رشته اتصال (DSN) 

        Returns:
            str: رشته اتصال برای ClickHouse
        """
        protocol = "https" if self.secure else "http"
        return f"{protocol}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    def get_servers(self) -> List[str]:
        """
        استخراج لیست سرورهای ClickHouse از تنظیمات

        پشتیبانی از چندین سرور با جداکننده کاما در متغیر محیطی CLICKHOUSE_HOST

        Returns:
            List[str]: لیست آدرس‌های سرورهای ClickHouse
        """
        return self.host.split(',') if ',' in self.host else [self.host]

    def get_circuit_breaker_config(self) -> Dict[str, int]:
        """
        دریافت تنظیمات Circuit Breaker

        Returns:
            Dict[str, int]: تنظیمات Circuit Breaker
        """
        return {
            "max_failures": self.circuit_breaker_max_failures,
            "reset_timeout": self.circuit_breaker_reset_timeout
        }

    def get_retry_config(self) -> Dict[str, int]:
        """
        دریافت تنظیمات Retry

        Returns:
            Dict[str, int]: تنظیمات مکانیزم Retry
        """
        return {
            "max_attempts": self.retry_max_attempts,
            "min_wait": self.retry_min_wait,
            "max_wait": self.retry_max_wait
        }

    def get_security_config(self) -> Dict[str, Any]:
        """
        دریافت تنظیمات امنیتی

        Returns:
            Dict[str, Any]: تنظیمات امنیتی
        """
        return {
            "access_control_secret": self.access_control_secret,
            "access_token_expiry": self.access_token_expiry,
            "encryption_key": self.encryption_key
        }

    def get_monitoring_config(self) -> Dict[str, int]:
        """
        دریافت تنظیمات مانیتورینگ

        Returns:
            Dict[str, int]: تنظیمات مانیتورینگ
        """
        return {
            "prometheus_port": self.prometheus_port,
            "monitoring_interval": self.monitoring_interval
        }

    def get_data_management_config(self) -> Dict[str, Any]:
        """
        دریافت تنظیمات مدیریت داده

        Returns:
            Dict[str, Any]: تنظیمات مدیریت داده
        """
        return {
            "retention_days": self.data_retention_days,
            "backup_interval": self.backup_interval,
            "backup_dir": self.backup_dir
        }


# ایجاد نمونه پیش‌فرض قابل استفاده در سراسر کد
config = ClickHouseConfig()

