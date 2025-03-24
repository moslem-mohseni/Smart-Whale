from typing import Dict, Any
import time
from .task_scheduler import TaskScheduler
from .optimizer import ProcessingOptimizer

class ProcessingPipeline:
    """
    ماژول مدیریت پردازش چندسطحی برای مدل‌های هوش مصنوعی.
    """

    def __init__(self):
        """
        مقداردهی اولیه و تنظیم ماژول‌های زمان‌بندی و بهینه‌سازی.
        """
        self.task_scheduler = TaskScheduler()
        self.optimizer = ProcessingOptimizer()

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        مدیریت و اجرای پردازش بر اساس سطح پیچیدگی درخواست.
        :param request: دیکشنری شامل اطلاعات ورودی.
        :return: نتیجه پردازش.
        """
        if "data" not in request or "complexity" not in request:
            raise ValueError("درخواست نامعتبر است! داده‌ی ورودی و سطح پیچیدگی باید مشخص باشند.")

        complexity = request["complexity"]
        allocated_resources = self.optimizer.allocate_resources(complexity)

        if not allocated_resources:
            return {"error": "عدم تخصیص منابع کافی برای پردازش!"}

        # زمان‌بندی وظایف پردازشی
        self.task_scheduler.schedule_task(request)

        # انتخاب سطح پردازش مناسب
        if complexity == "quick":
            result = self._quick_processing(request["data"])
        elif complexity == "normal":
            result = self._normal_processing(request["data"])
        elif complexity == "deep":
            result = self._deep_processing(request["data"])
        else:
            raise ValueError("سطح پیچیدگی پردازش نامعتبر است!")

        return {"result": result, "complexity": complexity, "processing_time": time.time()}

    def _quick_processing(self, data: Any) -> Any:
        """
        پردازش سریع برای درخواست‌های سبک.
        :param data: داده‌ی ورودی.
        :return: نتیجه پردازش سریع.
        """
        return f"Quick processed data: {data}"

    def _normal_processing(self, data: Any) -> Any:
        """
        پردازش استاندارد برای درخواست‌های معمولی.
        :param data: داده‌ی ورودی.
        :return: نتیجه پردازش استاندارد.
        """
        time.sleep(1)  # شبیه‌سازی پردازش با زمان استاندارد
        return f"Normal processed data: {data}"

    def _deep_processing(self, data: Any) -> Any:
        """
        پردازش عمیق برای درخواست‌های پیچیده.
        :param data: داده‌ی ورودی.
        :return: نتیجه پردازش عمیق.
        """
        time.sleep(2)  # شبیه‌سازی پردازش عمیق با زمان بیشتر
        return f"Deep processed data: {data}"
