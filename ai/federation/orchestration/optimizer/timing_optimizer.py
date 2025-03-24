from typing import Dict, Any, List
import time


class TimingOptimizer:
    """
    بهینه‌سازی زمان‌بندی اجرای وظایف برای کاهش تأخیر و افزایش کارایی
    """

    def __init__(self):
        self.task_timestamps: List[Dict[str, Any]] = []  # نگهداری تاریخچه اجرای وظایف

    def log_task_execution(self, model_name: str, task_type: str) -> None:
        """
        ثبت زمان اجرای یک وظیفه برای تحلیل بهینه‌سازی
        :param model_name: نام مدل
        :param task_type: نوع وظیفه پردازشی
        """
        self.task_timestamps.append({
            "model": model_name,
            "task": task_type,
            "timestamp": time.time()
        })

    def analyze_execution_timing(self) -> str:
        """
        تحلیل زمان‌بندی اجرای وظایف و ارائه پیشنهادات بهینه‌سازی
        :return: پیشنهادات بهینه‌سازی به صورت رشته
        """
        if len(self.task_timestamps) < 2:
            return "Not enough data for analysis."

        time_diffs = [
            self.task_timestamps[i]["timestamp"] - self.task_timestamps[i - 1]["timestamp"]
            for i in range(1, len(self.task_timestamps))
        ]
        avg_time_gap = sum(time_diffs) / len(time_diffs)

        if avg_time_gap > 10.0:
            return "Consider reducing task wait times or increasing parallel execution."
        elif avg_time_gap > 5.0:
            return "Optimize task scheduling to balance workload efficiently."
        else:
            return "Task execution timing is optimal."


# نمونه استفاده از TimingOptimizer برای تست
if __name__ == "__main__":
    optimizer = TimingOptimizer()
    optimizer.log_task_execution("model_a", "classification")
    time.sleep(6)
    optimizer.log_task_execution("model_b", "segmentation")
    time.sleep(4)
    optimizer.log_task_execution("model_c", "detection")

    print(f"Execution Timing Analysis: {optimizer.analyze_execution_timing()}")
