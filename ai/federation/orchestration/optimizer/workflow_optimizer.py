from typing import Dict, Any, List


class WorkflowOptimizer:
    """
    بهینه‌سازی روند اجرای پردازش‌ها برای افزایش بهره‌وری سیستم
    """

    def __init__(self):
        self.execution_logs: List[Dict[str, Any]] = []  # نگهداری تاریخچه اجرای پردازش‌ها

    def log_execution(self, model_name: str, task_type: str, duration: float) -> None:
        """
        ثبت اطلاعات اجرای پردازش برای تحلیل و بهینه‌سازی
        :param model_name: نام مدل
        :param task_type: نوع وظیفه پردازشی
        :param duration: مدت زمان اجرای پردازش (ثانیه)
        """
        self.execution_logs.append({
            "model": model_name,
            "task": task_type,
            "duration": duration
        })

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """
        دریافت تاریخچه اجرای پردازش‌ها
        :return: لیستی از پردازش‌های اجراشده
        """
        return self.execution_logs

    def suggest_workflow_improvement(self) -> str:
        """
        ارائه پیشنهادات برای بهینه‌سازی روند اجرای پردازش‌ها
        :return: پیشنهادات به صورت رشته
        """
        if not self.execution_logs:
            return "No execution data available for analysis."

        avg_duration = sum(log["duration"] for log in self.execution_logs) / len(self.execution_logs)

        if avg_duration > 4.0:
            return "Consider parallel execution or task scheduling optimization."
        elif avg_duration > 2.0:
            return "Optimize task dependencies to reduce bottlenecks."
        else:
            return "Workflow is performing efficiently."


# نمونه استفاده از WorkflowOptimizer برای تست
if __name__ == "__main__":
    optimizer = WorkflowOptimizer()
    optimizer.log_execution("model_a", "classification", 5.2)
    optimizer.log_execution("model_b", "segmentation", 3.1)
    optimizer.log_execution("model_c", "detection", 2.8)

    print(f"Execution History: {optimizer.get_execution_history()}")
    print(f"Workflow Improvement Suggestions: {optimizer.suggest_workflow_improvement()}")
