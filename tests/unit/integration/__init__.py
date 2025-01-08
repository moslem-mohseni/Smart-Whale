# tests/integration/__init__.py
"""
Integration Testing Module
-----------------------
این ماژول مسئول تست‌های یکپارچگی است که تعامل بین بخش‌های مختلف سیستم را بررسی می‌کند.

تست‌های یکپارچگی اطمینان حاصل می‌کنند که:
- بخش‌های مختلف سیستم به درستی با هم کار می‌کنند
- جریان داده بین سرویس‌ها صحیح است
- خطاها به درستی مدیریت می‌شوند
- عملکرد سیستم تحت شرایط واقعی قابل قبول است

نکات مهم در تست‌های یکپارچگی:
1. نیاز به راه‌اندازی محیط تست کامل
2. زمان اجرای طولانی‌تر نسبت به تست‌های واحد
3. نیاز به مدیریت وضعیت و پاکسازی بعد از هر تست
4. شبیه‌سازی شرایط واقعی تا حد امکان
"""

import pytest
import docker
from typing import Any, Dict, List
from pathlib import Path
from datetime import datetime


class IntegrationTestConfiguration:
    """تنظیمات پایه برای تست‌های یکپارچگی"""

    def __init__(self):
        self.docker_client = docker.from_env()
        self.containers = {}
        self.networks = {}
        self.test_data_path = Path(__file__).parent / 'data'
        self.timeout = 30  # زمان انتظار برای راه‌اندازی سرویس‌ها

    def cleanup(self):
        """پاکسازی منابع تست"""
        for container in self.containers.values():
            container.stop()
        for network in self.networks.values():
            network.remove()

    @staticmethod
    def create_test_environment(services: List[str]) -> Dict[str, Any]:
        """ایجاد محیط تست برای سرویس‌های مورد نیاز"""
        return {
            'name': f"test_env_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'services': services,
            'created_at': datetime.now()
        }


# tests/integration/infrastructure/__init__.py
"""
Infrastructure Integration Tests
-----------------------------
این ماژول شامل تست‌های یکپارچگی برای لایه زیرساخت است.

موارد تحت پوشش:
- تست سرویس‌های پایگاه داده (TimescaleDB و ClickHouse)
- تست سیستم کش (Redis)
- تست سیستم پیام‌رسانی (Kafka)
- تست عملکرد همزمان سرویس‌ها
"""

from dataclasses import dataclass
from typing import Dict, Any
from datetime import datetime


@dataclass
class ServiceTestConfig:
    """تنظیمات تست برای هر سرویس"""
    name: str
    port: int
    environment: Dict[str, str]
    healthcheck_endpoint: str
    startup_timeout: int = 30

    def get_container_config(self) -> Dict[str, Any]:
        """تولید تنظیمات کانتینر برای سرویس"""
        return {
            'name': f"test_{self.name}",
            'ports': {f"{self.port}/tcp": self.port},
            'environment': self.environment,
            'detach': True,
            'remove': True
        }