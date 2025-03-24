import os
from dataclasses import dataclass
from typing import Optional, List, Dict
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RedisConfig:
    """
    تنظیمات اتصال به Redis
    """
    host: str = os.getenv('REDIS_HOST', 'localhost')
    port: int = int(os.getenv('REDIS_PORT', 6379))
    database: int = int(os.getenv('REDIS_DB', 0))
    password: Optional[str] = os.getenv('REDIS_PASSWORD', None)

    max_connections: int = int(os.getenv('REDIS_MAX_CONNECTIONS', 10))
    socket_timeout: int = int(os.getenv('REDIS_SOCKET_TIMEOUT', 5))
    socket_connect_timeout: int = int(os.getenv('REDIS_CONNECT_TIMEOUT', 5))
    retry_on_timeout: bool = os.getenv('REDIS_RETRY_ON_TIMEOUT', 'True').lower() == 'true'

    ssl: bool = os.getenv('REDIS_SSL', 'False').lower() == 'true'
    ssl_cert_reqs: Optional[str] = os.getenv('REDIS_SSL_CERT_REQS')
    ssl_certfile: Optional[str] = os.getenv('REDIS_SSL_CERTFILE')
    ssl_keyfile: Optional[str] = os.getenv('REDIS_SSL_KEYFILE')
    ssl_ca_certs: Optional[str] = os.getenv('REDIS_SSL_CA_CERTS')

    cluster_mode: bool = os.getenv('REDIS_CLUSTER_MODE', 'False').lower() == 'true'
    cluster_nodes: Optional[List[Dict[str, str]]] = None

    sentinel_mode: bool = os.getenv('REDIS_SENTINEL_MODE', 'False').lower() == 'true'
    sentinel_master: Optional[str] = os.getenv('REDIS_SENTINEL_MASTER')
    sentinel_nodes: Optional[List[Dict[str, str]]] = None

    def get_connection_params(self) -> dict:
        """تولید پارامترهای اتصال برای aioredis"""
        params = {
            'host': self.host,
            'port': self.port,
            'db': self.database,
            'password': self.password,
            'maxsize': self.max_connections,
            'timeout': self.socket_timeout,
            'retry_on_timeout': self.retry_on_timeout
        }

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
        """تولید پارامترهای اتصال برای حالت کلاستر"""
        if not self.cluster_mode:
            raise ValueError("حالت کلاستر فعال نیست")

        return {
            'startup_nodes': self.cluster_nodes,
            'password': self.password,
            'ssl': self.ssl
        }

    def get_sentinel_params(self) -> dict:
        """تولید پارامترهای اتصال برای حالت Sentinel"""
        if not self.sentinel_mode:
            raise ValueError("حالت Sentinel فعال نیست")

        return {
            'master_name': self.sentinel_master,
            'sentinels': self.sentinel_nodes,
            'password': self.password,
            'db': self.database,
            'ssl': self.ssl
        }

    @classmethod
    def from_env(cls) -> 'RedisConfig':
        """ایجاد نمونه از کانفیگ با استفاده از متغیرهای محیطی"""
        return cls()


# نصب بسته مورد نیاز:
# pip install python-dotenv

