from typing import Dict, Any, List
from core.scheduler.task_scheduler import TaskScheduler
from core.scheduler.resource_allocator import ResourceAllocator
from core.scheduler.priority_manager import PriorityManager
from core.scheduler.dependency_resolver import DependencyResolver
from core.resource_management.monitor.resource_monitor import ResourceMonitor
from core.resilience.circuit_breaker.breaker_manager import BreakerManager

class OperationCoordinator:
    """
    هماهنگ‌کننده عملیات پردازشی برای مدیریت اجرای وظایف و تخصیص منابع.
    """

    def __init__(self):
        """
        مقداردهی اولیه هماهنگ‌کننده عملیات‌ها.
        """
        self.task_scheduler = TaskScheduler()
        self.resource_allocator = ResourceAllocator()
        self.priority_manager = PriorityManager()
        self.dependency_resolver = DependencyResolver()
        self.resource_monitor = ResourceMonitor()
        self.breaker_manager = BreakerManager()

    def execute_task(self, task_id: str, task_func: callable, cpu_demand: float, memory_demand: float, priority: int = 1) -> Dict[str, Any]:
        """
        اجرای یک وظیفه با هماهنگی بین منابع و اولویت‌ها.

        :param task_id: شناسه وظیفه
        :param task_func: تابع مربوط به وظیفه
        :param cpu_demand: میزان نیاز به CPU (0 تا 1)
        :param memory_demand: میزان نیاز به حافظه (0 تا 1)
        :param priority: سطح اولویت اجرای وظیفه
        :return: دیکشنری شامل وضعیت اجرای وظیفه
        """
        # بررسی وضعیت قطع‌کننده مدار
        if self.breaker_manager.is_circuit_open():
            return {"status": "denied", "reason": "Circuit breaker is active"}

        # بررسی وابستگی‌های انجام‌نشده
        if not self.dependency_resolver.can_execute(task_id):
            return {"status": "denied", "reason": "Unresolved dependencies"}

        # بررسی منابع قبل از اجرا
        allocation = self.resource_allocator.allocate_resources(cpu_demand, memory_demand)
        if allocation["status"] == "denied":
            return allocation  # تخصیص منابع انجام نشد

        # تنظیم اولویت
        self.priority_manager.set_priority(task_id, priority)

        # زمان‌بندی و اجرای وظیفه
        self.task_scheduler.schedule_task(task_func, priority, 0)
        self.task_scheduler.run_scheduler()

        return {"status": "executed", "task_id": task_id}

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        دریافت وضعیت یک وظیفه مشخص.

        :param task_id: شناسه وظیفه
        :return: دیکشنری شامل اطلاعات وظیفه
        """
        priority = self.priority_manager.get_priority(task_id)
        dependencies = self.dependency_resolver.get_dependencies(task_id)

        return {
            "task_id": task_id,
            "priority": priority,
            "dependencies": dependencies,
            "resource_status": self.resource_monitor.get_system_resources()
        }
