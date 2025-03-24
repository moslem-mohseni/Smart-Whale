from typing import Dict, Any
from core.scheduler.resource_allocator import ResourceAllocator
from core.scheduler.priority_manager import PriorityManager
from core.resource_management.monitor.resource_monitor import ResourceMonitor
from core.resilience.circuit_breaker.breaker_manager import BreakerManager

class ModelCoordinator:
    """
    هماهنگ‌کننده مدل‌ها برای مدیریت تخصیص منابع و زمان‌بندی اجرای مدل‌های پردازشی.
    """

    def __init__(self):
        """
        مقداردهی اولیه هماهنگ‌کننده مدل‌ها.
        """
        self.resource_allocator = ResourceAllocator()
        self.priority_manager = PriorityManager()
        self.resource_monitor = ResourceMonitor()
        self.breaker_manager = BreakerManager()
        self.model_registry: Dict[str, Dict[str, Any]] = {}  # اطلاعات ثبت‌شده مدل‌ها

    def register_model(self, model_id: str, cpu_demand: float, memory_demand: float, priority: int = 1) -> None:
        """
        ثبت مدل برای هماهنگ‌سازی منابع.

        :param model_id: شناسه مدل
        :param cpu_demand: میزان نیاز به CPU (0 تا 1)
        :param memory_demand: میزان نیاز به حافظه (0 تا 1)
        :param priority: سطح اولویت اجرای مدل
        """
        self.model_registry[model_id] = {
            "cpu_demand": cpu_demand,
            "memory_demand": memory_demand,
            "priority": priority,
            "status": "registered"
        }
        self.priority_manager.set_priority(model_id, priority)

    def allocate_resources_for_model(self, model_id: str) -> Dict[str, Any]:
        """
        تخصیص منابع به یک مدل خاص.

        :param model_id: شناسه مدل
        :return: دیکشنری شامل وضعیت تخصیص منابع
        """
        if model_id not in self.model_registry:
            return {"status": "error", "reason": "Model not registered"}

        model_info = self.model_registry[model_id]

        # بررسی قطع‌کننده مدار قبل از تخصیص منابع
        if self.breaker_manager.is_circuit_open():
            return {"status": "denied", "reason": "Circuit breaker is active"}

        allocation = self.resource_allocator.allocate_resources(
            model_info["cpu_demand"],
            model_info["memory_demand"]
        )

        if allocation["status"] == "allocated":
            self.model_registry[model_id]["status"] = "running"
        return allocation

    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """
        دریافت وضعیت یک مدل خاص.

        :param model_id: شناسه مدل
        :return: دیکشنری شامل اطلاعات وضعیت مدل
        """
        if model_id not in self.model_registry:
            return {"status": "error", "reason": "Model not found"}

        model_info = self.model_registry[model_id]
        return {
            "model_id": model_id,
            "status": model_info["status"],
            "priority": model_info["priority"],
            "resource_status": self.resource_monitor.get_system_resources()
        }

    def release_resources(self, model_id: str) -> None:
        """
        آزادسازی منابع اختصاص داده شده به یک مدل.

        :param model_id: شناسه مدل
        """
        if model_id in self.model_registry:
            self.model_registry[model_id]["status"] = "stopped"
