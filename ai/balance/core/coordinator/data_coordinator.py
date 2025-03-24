from typing import Dict, Any
from core.resource_management.monitor.resource_monitor import ResourceMonitor
from core.scheduler.task_scheduler import TaskScheduler
from core.resilience.circuit_breaker.breaker_manager import BreakerManager
from core.monitoring.metrics.collector import MetricsCollector

class DataCoordinator:
    """
    هماهنگ‌کننده داده‌ها برای مدیریت تخصیص داده و همگام‌سازی بین منابع.
    """

    def __init__(self):
        """
        مقداردهی اولیه هماهنگ‌کننده داده‌ها.
        """
        self.resource_monitor = ResourceMonitor()
        self.task_scheduler = TaskScheduler()
        self.breaker_manager = BreakerManager()
        self.metrics_collector = MetricsCollector()
        self.data_registry: Dict[str, Dict[str, Any]] = {}  # ثبت داده‌های قابل دسترسی

    def register_data_source(self, data_id: str, size: int, source: str) -> None:
        """
        ثبت یک منبع داده برای مدیریت و هماهنگی.

        :param data_id: شناسه داده
        :param size: اندازه داده (بر حسب بایت)
        :param source: منبع داده (مثلاً "clickhouse" یا "redis")
        """
        self.data_registry[data_id] = {
            "size": size,
            "source": source,
            "status": "available"
        }
        self.metrics_collector.collect_metric("data_registered", {"data_id": data_id, "size": size, "source": source})

    def allocate_data(self, data_id: str) -> Dict[str, Any]:
        """
        تخصیص داده به یک پردازش یا مدل.

        :param data_id: شناسه داده
        :return: دیکشنری شامل وضعیت تخصیص داده
        """
        if data_id not in self.data_registry:
            return {"status": "error", "reason": "Data not found"}

        # بررسی وضعیت قطع‌کننده مدار قبل از تخصیص داده
        if self.breaker_manager.is_circuit_open():
            return {"status": "denied", "reason": "Circuit breaker is active"}

        data_info = self.data_registry[data_id]
        if data_info["status"] == "allocated":
            return {"status": "denied", "reason": "Data already in use"}

        # بررسی منابع سیستم قبل از تخصیص داده
        system_resources = self.resource_monitor.get_system_resources()
        if system_resources["memory_usage"] > 0.85:
            return {"status": "denied", "reason": "Insufficient memory for data allocation"}

        # تخصیص داده
        self.data_registry[data_id]["status"] = "allocated"
        self.metrics_collector.collect_metric("data_allocated", {"data_id": data_id, "size": data_info["size"]})

        return {"status": "allocated", "data_id": data_id, "source": data_info["source"]}

    def release_data(self, data_id: str) -> None:
        """
        آزادسازی داده‌های اختصاص یافته.

        :param data_id: شناسه داده
        """
        if data_id in self.data_registry:
            self.data_registry[data_id]["status"] = "available"
            self.metrics_collector.collect_metric("data_released", {"data_id": data_id})

    def synchronize_data(self, data_id: str, destination: str) -> Dict[str, Any]:
        """
        همگام‌سازی داده بین منابع مختلف.

        :param data_id: شناسه داده
        :param destination: مقصدی که داده باید همگام‌سازی شود
        :return: دیکشنری شامل وضعیت همگام‌سازی
        """
        if data_id not in self.data_registry:
            return {"status": "error", "reason": "Data not found"}

        data_info = self.data_registry[data_id]

        # بررسی منابع قبل از همگام‌سازی
        system_resources = self.resource_monitor.get_system_resources()
        if system_resources["cpu_usage"] > 0.9:
            return {"status": "denied", "reason": "High CPU usage, cannot sync data"}

        # شبیه‌سازی فرآیند همگام‌سازی
        self.task_scheduler.schedule_task(self._sync_task, priority=2, delay=0, data_id=data_id, destination=destination)

        self.metrics_collector.collect_metric("data_synchronized", {"data_id": data_id, "destination": destination})

        return {"status": "synchronizing", "data_id": data_id, "destination": destination}

    def _sync_task(self, data_id: str, destination: str) -> None:
        """
        وظیفه همگام‌سازی داده (اجرای واقعی در زمان مناسب).
        """
        if data_id in self.data_registry:
            self.data_registry[data_id]["status"] = "synchronized"
