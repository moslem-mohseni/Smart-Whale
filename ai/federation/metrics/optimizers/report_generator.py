from typing import Dict


class ReportGenerator:
    """
    تولید گزارش‌های تحلیل متریک‌های عملکردی سیستم
    """

    def __init__(self):
        self.reports: Dict[str, str] = {}  # ذخیره گزارش‌های تولیدشده

    def generate_report(self, model_name: str, metrics: Dict[str, float]) -> str:
        """
        تولید گزارش از متریک‌های یک مدل مشخص‌شده
        :param model_name: نام مدل
        :param metrics: دیکشنری شامل متریک‌های مدل
        :return: گزارش به‌صورت رشته
        """
        report = f"Report for {model_name}:\n"
        report += "\n".join([f"{metric}: {value}" for metric, value in metrics.items()])

        self.reports[model_name] = report
        return report

    def get_report(self, model_name: str) -> str:
        """
        دریافت گزارش یک مدل خاص
        :param model_name: نام مدل
        :return: گزارش مدل یا پیام پیش‌فرض در صورت نبود گزارش
        """
        return self.reports.get(model_name, "No report available for this model.")

    def get_all_reports(self) -> Dict[str, str]:
        """
        دریافت تمام گزارش‌های ثبت‌شده
        :return: دیکشنری شامل گزارش‌های تمام مدل‌ها
        """
        return self.reports


# نمونه استفاده از ReportGenerator برای تست
if __name__ == "__main__":
    report_generator = ReportGenerator()
    metrics = {"accuracy": 0.92, "latency": 150.3, "throughput": 32.1}
    report = report_generator.generate_report("model_a", metrics)
    print(report)
    print(f"Stored Report: {report_generator.get_report('model_a')}")
