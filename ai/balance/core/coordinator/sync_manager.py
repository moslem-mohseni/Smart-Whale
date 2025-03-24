from typing import Dict, Any
from core.resource_management.monitor.resource_monitor import ResourceMonitor
from core.scheduler.task_scheduler import TaskScheduler
from core.resilience.circuit_breaker.breaker_manager import BreakerManager
from core.monitoring.metrics.collector import MetricsCollector

class SyncManager:
    """
    مدیریت همگام‌سازی داده‌ها و وضعیت‌ها بین ماژول‌های مختلف سیستم.
    """

    def __init__(self):
        """
        مقداردهی اولیه مدیر همگام‌سازی.
        """
        self.resource_monitor = ResourceMonitor()
        self.task_scheduler = TaskScheduler()
        self.breaker_manager = BreakerManager()
        self.metrics_collector = MetricsCollector()
        self.sync_registry: Dict[str, Dict[str, Any]] = {}  # اطلاعات ثبت‌شده همگام‌سازی‌ها

    def register_sync_task(self, sync_id: str, source: str, destination: str, size: int) -> None:
        """
        ثبت یک فرآیند همگام‌سازی.

        :param sync_id: شناسه همگام‌سازی
        :param source: منبع داده
        :param destination: مقصد داده
        :param size: اندازه داده برای همگام‌سازی (برحسب بایت)
        """
        self.sync_registry[sync_id] = {
            "source": source,
            "destination": destination,
            "size": size,
            "status": "pending"
        }
        self.metrics_collector.collect_metric("sync_registered", {"sync_id": sync_id, "size": size, "source": source, "destination": destination})

    def can_sync(self, sync_id: str) -> bool:
        """
        بررسی اینکه آیا امکان همگام‌سازی وجود دارد.

        :param sync_id: شناسه همگام‌سازی
        :return: `True` اگر منابع کافی برای همگام‌سازی وجود داشته باشد، `False` در غیر این صورت
        """
        if sync_id not in self.sync_registry:
            return False

        # بررسی منابع قبل از همگام‌سازی
        system_resources = self.resource_monitor.get_system_resources()
        if system_resources["cpu_usage"] > 0.9 or system_resources["memory_usage"] > 0.85:
            return False

        # بررسی وضعیت قطع‌کننده مدار
        if self.breaker_manager.is_circuit_open():
            return False  # جلوگیری از پردازش در شرایط بحرانی

        return True

    def execute_sync(self, sync_id: str) -> Dict[str, Any]:
        """
        اجرای فرآیند همگام‌سازی در صورت فراهم بودن منابع.

        :param sync_id: شناسه همگام‌سازی
        :return: دیکشنری شامل وضعیت فرآیند همگام‌سازی
        """
        if sync_id not in self.sync_registry:
            return {"status": "error", "reason": "Sync task not found"}

        if not self.can_sync(sync_id):
            return {"status": "denied", "reason": "Insufficient system resources or circuit breaker active"}

        sync_info = self.sync_registry[sync_id]

        # زمان‌بندی وظیفه همگام‌سازی
        self.task_scheduler.schedule_task(self._sync_task, priority=2, delay=0, sync_id=sync_id)

        self.metrics_collector.collect_metric("sync_started", {"sync_id": sync_id, "source": sync_info["source"], "destination": sync_info["destination"]})

        return {"status": "synchronizing", "sync_id": sync_id}

    def _sync_task(self, sync_id: str) -> None:
        """
        وظیفه واقعی همگام‌سازی داده که در زمان مناسب اجرا می‌شود.
        """
        if sync_id in self.sync_registry:
            self.sync_registry[sync_id]["status"] = "completed"
            self.metrics_collector.collect_metric("sync_completed", {"sync_id": sync_id})

    def get_sync_status(self, sync_id: str) -> Dict[str, Any]:
        """
        دریافت وضعیت یک فرآیند همگام‌سازی.

        :param sync_id: شناسه همگام‌سازی
        :return: دیکشنری شامل وضعیت همگام‌سازی
        """
        if sync_id not in self.sync_registry:
            return {"status": "error", "reason": "Sync task not found"}

        return self.sync_registry[sync_id]
