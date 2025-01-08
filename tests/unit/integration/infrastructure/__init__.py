# integration/infrastructure/__init__.py

"""
Infrastructure Integration Test Module
-----------------------------------
این ماژول تست‌های یکپارچگی زیرساخت را مدیریت می‌کند و اطمینان حاصل می‌کند که
تمام سرویس‌های زیرساختی به درستی با یکدیگر کار می‌کنند.

چهار سرویس اصلی که در این تست‌ها بررسی می‌شوند:
1. ClickHouse: برای تحلیل داده‌ها (test_analytics_service.py)
2. Redis: برای مدیریت کش (test_cache_service.py)
3. Kafka: برای پیام‌رسانی (test_messaging_service.py)
4. TimescaleDB: برای ذخیره‌سازی داده‌های سری زمانی (test_storage_service.py)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import docker
import asyncio


@dataclass
class ServiceContainer:
    """مدیریت اطلاعات کانتینر هر سرویس"""
    name: str
    container_id: str
    port_mappings: Dict[str, int]
    status: str
    health_check_url: Optional[str] = None


class InfrastructureTestEnvironment:
    """
    مدیریت محیط تست یکپارچگی زیرساخت

    این کلاس مسئول راه‌اندازی، نظارت و پاکسازی تمام سرویس‌های مورد نیاز
    برای تست‌های یکپارچگی است.
    """

    def __init__(self):
        self.docker_client = docker.from_env()
        self.containers: Dict[str, ServiceContainer] = {}
        self.network_name = "infra_test_network"

    async def setup(self) -> None:
        """راه‌اندازی تمام سرویس‌های مورد نیاز برای تست"""
        # ایجاد شبکه داکر برای ارتباط سرویس‌ها
        self.network = self.docker_client.networks.create(
            self.network_name, driver="bridge"
        )

        # راه‌اندازی سرویس‌ها به ترتیب وابستگی
        await self._start_cache_service()  # Redis
        await self._start_messaging_service()  # Kafka
        await self._start_storage_service()  # TimescaleDB
        await self._start_analytics_service()  # ClickHouse

    async def teardown(self) -> None:
        """پاکسازی و حذف تمام سرویس‌ها"""
        for container in self.containers.values():
            self.docker_client.containers.get(container.container_id).stop()

        if hasattr(self, 'network'):
            self.network.remove()

    async def verify_services_health(self) -> Dict[str, bool]:
        """بررسی سلامت تمام سرویس‌ها"""
        results = {}
        for service in self.containers.values():
            if service.health_check_url:
                # بررسی سلامت سرویس
                healthy = await self._check_service_health(service)
                results[service.name] = healthy
        return results


# Export کلاس‌های مورد نیاز
__all__ = [
    'InfrastructureTestEnvironment',
    'ServiceContainer'
]