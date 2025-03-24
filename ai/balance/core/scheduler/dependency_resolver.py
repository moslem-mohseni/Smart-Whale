from typing import Dict, List, Tuple
from core.resilience.circuit_breaker.breaker_manager import BreakerManager
from core.resource_management.monitor.resource_monitor import ResourceMonitor

class DependencyResolver:
    """
    مدیریت و حل وابستگی‌های بین وظایف پردازشی.
    """

    def __init__(self):
        """
        مقداردهی اولیه سیستم حل وابستگی‌ها.
        """
        self.task_dependencies: Dict[str, List[str]] = {}  # وابستگی‌ها بین وظایف
        self.breaker_manager = BreakerManager()
        self.resource_monitor = ResourceMonitor()

    def add_dependency(self, task_id: str, dependencies: List[str]) -> None:
        """
        اضافه کردن وابستگی‌های یک وظیفه.

        :param task_id: شناسه وظیفه
        :param dependencies: لیستی از شناسه‌های وظایف وابسته
        """
        self.task_dependencies[task_id] = dependencies

    def get_dependencies(self, task_id: str) -> List[str]:
        """
        دریافت لیست وابستگی‌های یک وظیفه.

        :param task_id: شناسه وظیفه
        :return: لیست وابستگی‌های وظیفه
        """
        return self.task_dependencies.get(task_id, [])

    def can_execute(self, task_id: str) -> bool:
        """
        بررسی امکان اجرای یک وظیفه با توجه به وابستگی‌ها و وضعیت منابع.

        :param task_id: شناسه وظیفه
        :return: `True` اگر وظیفه قابل اجرا باشد، `False` در غیر این صورت
        """
        # بررسی وضعیت قطع‌کننده مدار
        if self.breaker_manager.is_circuit_open():
            return False  # اجرای وظیفه متوقف می‌شود

        # بررسی منابع سیستم
        system_resources = self.resource_monitor.get_system_resources()
        if system_resources["cpu_usage"] > 0.9 or system_resources["memory_usage"] > 0.95:
            return False  # اجرای وظیفه متوقف می‌شود

        # بررسی وابستگی‌های انجام‌نشده
        for dependency in self.task_dependencies.get(task_id, []):
            if dependency not in self.task_dependencies:
                return False  # وابستگی انجام نشده است

        return True  # همه شرایط مهیا است

    def resolve_dependencies(self, task_id: str) -> List[str]:
        """
        حل وابستگی‌ها و بازگرداندن لیست وظایف لازم‌الاجرا قبل از این وظیفه.

        :param task_id: شناسه وظیفه
        :return: لیستی از وظایف مورد نیاز برای اجرا
        """
        resolved_tasks = []
        dependencies = self.get_dependencies(task_id)

        for dependency in dependencies:
            if self.can_execute(dependency):
                resolved_tasks.append(dependency)

        return resolved_tasks
