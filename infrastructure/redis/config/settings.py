# infrastructure/redis/config/settings.py

from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class RedisConfig:
    """
    تنظیمات اتصال به Redis

    این کلاس تمام پارامترهای مورد نیاز برای اتصال و پیکربندی Redis را نگهداری می‌کند.
    پشتیبانی از حالت‌های مختلف اتصال (تکی، کلاستر، sentinel) را فراهم می‌کند.
    """
    # تنظیمات پایه
    host: str
    port: int
    database: int = 0
    password: Optional[str] = None

    # تنظیمات اتصال
    max_connections: int = 10
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True

    # تنظیمات SSL/TLS
    ssl: bool = False
    ssl_cert_reqs: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    ssl_ca_certs: Optional[str] = None

    # تنظیمات کلاستر
    cluster_mode: bool = False
    cluster_nodes: Optional[List[Dict[str, str]]] = None

    # تنظیمات Sentinel
    sentinel_mode: bool = False
    sentinel_master: Optional[str] = None
    sentinel_nodes: Optional[List[Dict[str, str]]] = None

    def get_connection_params(self) -> dict:
        """
        تولید پارامترهای اتصال برای aioredis

        Returns:
            دیکشنری حاوی پارامترهای اتصال
        """
        params = {
            'host': self.host,
            'port': self.port,
            'db': self.database,
            'password': self.password,
            'maxsize': self.max_connections,
            'timeout': self.socket_timeout,
            'retry_on_timeout': self.retry_on_timeout
        }

        # اضافه کردن تنظیمات SSL
        if self.ssl:
            params.update({
                'ssl': True,
                'ssl_cert_reqs': self.ssl_cert_reqs,
                'ssl_certfile': self.ssl_certfile,
                'ssl_keyfile': self.ssl_keyfile,
                'ssl_ca_certs': self.ssl_ca_certs
            })

        return params

    def get_cluster_params(self) -> dict:
        """
        تولید پارامترهای اتصال برای حالت کلاستر

        Returns:
            دیکشنری حاوی پارامترهای کلاستر
        """
        if not self.cluster_mode:
            raise ValueError("Cluster mode is not enabled")

        return {
            'startup_nodes': self.cluster_nodes,
            'password': self.password,
            'ssl': self.ssl
        }