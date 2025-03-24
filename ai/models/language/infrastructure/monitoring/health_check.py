import logging
from ai.core.monitoring.health.checker import HealthChecker
from ai.core.monitoring.health.reporter import HealthReporter
from ai.core.monitoring.metrics.collector import MetricsCollector
from infrastructure.monitoring.health_service import HealthService

class HealthCheck:
    """
    این کلاس وظیفه‌ی بررسی سلامت سرویس‌های زیرساختی پردازش زبان را بر عهده دارد.
    """

    def __init__(self, health_checker: HealthChecker, health_reporter: HealthReporter, metrics_collector: MetricsCollector, health_service: HealthService):
        self.health_checker = health_checker
        self.health_reporter = health_reporter
        self.metrics_collector = metrics_collector
        self.health_service = health_service
        logging.info("✅ HealthCheck مقداردهی شد و ارتباط با سرویس‌های مانیتورینگ برقرار شد.")

    async def check_service_health(self, service_name: str) -> dict:
        """
        بررسی سلامت یک سرویس خاص.

        :param service_name: نام سرویس مورد بررسی
        :return: گزارش سلامت سرویس
        """
        try:
            health_status = await self.health_checker.check(service_name)
            self.health_reporter.report(service_name, health_status)
            logging.info(f"📊 وضعیت سلامت سرویس `{service_name}` بررسی شد: {health_status}")
            return health_status
        except Exception as e:
            logging.error(f"❌ خطا در بررسی سلامت سرویس `{service_name}`: {e}")
            return {"status": "error", "message": str(e)}

    async def check_all_services(self) -> dict:
        """
        بررسی سلامت تمام سرویس‌های زیرساختی مرتبط با پردازش زبان.

        :return: دیکشنری شامل وضعیت سلامت همه‌ی سرویس‌ها
        """
        try:
            all_services_status = await self.health_service.get_all_services_status()
            for service, status in all_services_status.items():
                self.health_reporter.report(service, status)
            logging.info(f"📊 وضعیت سلامت تمام سرویس‌ها بررسی شد.")
            return all_services_status
        except Exception as e:
            logging.error(f"❌ خطا در بررسی سلامت تمام سرویس‌ها: {e}")
            return {"status": "error", "message": str(e)}
