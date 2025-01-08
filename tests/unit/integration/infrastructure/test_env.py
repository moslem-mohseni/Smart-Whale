# tests/integration/infrastructure/test_env.py

import docker
import asyncio
from typing import Dict, Any, Optional
import logging


class TestEnvironmentManager:
    """مدیر محیط تست برای راه‌اندازی و مدیریت سرویس‌های مورد نیاز

    این کلاس مسئول مدیریت کانتینرهای داکر، شبکه‌ها و تنظیمات محیط تست است.
    از این کلاس برای شبیه‌سازی محیط واقعی در تست‌های یکپارچگی استفاده می‌شود.
    """

    def __init__(self):
        """مقداردهی اولیه مدیر محیط تست"""
        self.client = docker.from_env()
        self.containers: Dict[str, Any] = {}
        self.network = None
        self.network_name = "analytics_test_net"
        self.logger = logging.getLogger(__name__)

    async def setup(self) -> Dict[str, Dict[str, Any]]:
        """راه‌اندازی محیط تست

        این متد شامل:
        1. پاکسازی محیط قبلی
        2. ایجاد شبکه جدید
        3. راه‌اندازی کانتینرهای مورد نیاز
        4. آماده‌سازی تنظیمات سرویس‌ها
        """
        await self._cleanup_previous_environment()
        await self._create_network()
        await self._start_containers()

        # انتظار برای آماده شدن سرویس‌ها
        await asyncio.sleep(10)

        return self._get_service_configs()

    async def _cleanup_previous_environment(self):
        """پاکسازی محیط قبلی در صورت وجود"""
        try:
            # حذف شبکه قبلی
            old_network = self.client.networks.get(self.network_name)
            old_network.remove()
            self.logger.info("شبکه قبلی با موفقیت حذف شد")
        except docker.errors.NotFound:
            self.logger.debug("شبکه قبلی یافت نشد")

        # حذف کانتینرهای قبلی
        services = ['clickhouse', 'redis']
        for service in services:
            try:
                container = self.client.containers.get(f'test_{service}')
                container.stop()
                container.remove()
                self.logger.info(f"کانتینر {service} قبلی با موفقیت حذف شد")
            except docker.errors.NotFound:
                self.logger.debug(f"کانتینر {service} قبلی یافت نشد")

    async def _create_network(self):
        """ایجاد شبکه داکر برای تست‌ها"""
        self.network = self.client.networks.create(
            self.network_name,
            driver="bridge"
        )
        self.logger.info("شبکه جدید با موفقیت ایجاد شد")

    async def _start_containers(self):
        """راه‌اندازی کانتینرهای مورد نیاز"""
        # راه‌اندازی ClickHouse
        self.containers['clickhouse'] = self.client.containers.run(
            'yandex/clickhouse-server:latest',
            ports={'8123/tcp': 8123, '9000/tcp': 9000},
            environment={
                'CLICKHOUSE_USER': 'test_user',
                'CLICKHOUSE_PASSWORD': 'test_pass',
                'CLICKHOUSE_DB': 'test_db'
            },
            detach=True,
            remove=True,
            name='test_clickhouse',
            network=self.network_name
        )
        self.logger.info("کانتینر ClickHouse با موفقیت راه‌اندازی شد")

        # راه‌اندازی Redis
        self.containers['redis'] = self.client.containers.run(
            'redis:6.2-alpine',
            ports={'6379/tcp': 6379},
            detach=True,
            remove=True,
            name='test_redis',
            network=self.network_name
        )
        self.logger.info("کانتینر Redis با موفقیت راه‌اندازی شد")

    def _get_service_configs(self) -> Dict[str, Dict[str, Any]]:
        """دریافت تنظیمات سرویس‌ها برای استفاده در تست‌ها"""
        return {
            'clickhouse': {
                'host': 'localhost',
                'port': 9000,
                'database': 'test_db',
                'user': 'test_user',
                'password': 'test_pass'
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'database': 0
            }
        }

    async def start_service(self, service_name: str):
        """راه‌اندازی مجدد یک سرویس خاص

        Args:
            service_name: نام سرویس ('redis' یا 'clickhouse')
        """
        if service_name not in self.containers:
            raise ValueError(f"سرویس {service_name} وجود ندارد")

        container = self.containers[service_name]
        container.start()
        self.logger.info(f"سرویس {service_name} مجدداً راه‌اندازی شد")

        # انتظار برای آماده شدن سرویس
        await asyncio.sleep(2)

    async def stop_service(self, service_name: str):
        """توقف یک سرویس خاص

        Args:
            service_name: نام سرویس ('redis' یا 'clickhouse')
        """
        if service_name not in self.containers:
            raise ValueError(f"سرویس {service_name} وجود ندارد")

        container = self.containers[service_name]
        container.stop()
        self.logger.info(f"سرویس {service_name} متوقف شد")

    async def cleanup(self):
        """پاکسازی کامل محیط تست"""
        for name, container in self.containers.items():
            try:
                container.stop()
                self.logger.info(f"کانتینر {name} با موفقیت متوقف شد")
            except Exception as e:
                self.logger.error(f"خطا در توقف کانتینر {name}: {str(e)}")

        if self.network:
            try:
                self.network.remove()
                self.logger.info("شبکه با موفقیت حذف شد")
            except Exception as e:
                self.logger.error(f"خطا در حذف شبکه: {str(e)}")

        self.containers.clear()

    async def get_container_logs(self, service_name: str) -> str:
        """دریافت لاگ‌های یک سرویس خاص

        Args:
            service_name: نام سرویس ('redis' یا 'clickhouse')

        Returns:
            str: لاگ‌های سرویس
        """
        if service_name not in self.containers:
            raise ValueError(f"سرویس {service_name} وجود ندارد")

        container = self.containers[service_name]
        return container.logs().decode('utf-8')

    async def container_health_check(self, service_name: str) -> bool:
        """بررسی سلامت یک سرویس خاص

        Args:
            service_name: نام سرویس ('redis' یا 'clickhouse')

        Returns:
            bool: وضعیت سلامت سرویس
        """
        if service_name not in self.containers:
            raise ValueError(f"سرویس {service_name} وجود ندارد")

        container = self.containers[service_name]
        inspection = container.inspect()
        return inspection['State']['Running']