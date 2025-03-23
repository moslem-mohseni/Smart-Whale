from typing import List, Dict
from collections import Counter


class PatternAnalyzer:
    """
    تحلیل الگوهای درخواست برای پیش‌بینی روندهای آینده در بار پردازشی
    """

    def __init__(self, history_size: int = 20):
        self.request_history = []  # ذخیره درخواست‌های گذشته
        self.history_size = history_size

    def record_request(self, request_type: str) -> None:
        """
        ثبت نوع درخواست در تاریخچه برای تحلیل الگوها
        :param request_type: نوع درخواست (مانند classification, regression)
        """
        if len(self.request_history) >= self.history_size:
            self.request_history.pop(0)  # حذف قدیمی‌ترین درخواست برای حفظ ظرفیت ثابت
        self.request_history.append(request_type)

    def analyze_patterns(self) -> Dict[str, float]:
        """
        تحلیل توزیع فراوانی انواع درخواست‌ها و ارائه پیش‌بینی بر اساس روندهای مشاهده‌شده
        :return: دیکشنری شامل توزیع درصدی درخواست‌های اخیر
        """
        total_requests = len(self.request_history)
        if total_requests == 0:
            return {}

        request_counts = Counter(self.request_history)
        return {req: round(count / total_requests * 100, 2) for req, count in request_counts.items()}


# نمونه استفاده از PatternAnalyzer برای تست
if __name__ == "__main__":
    analyzer = PatternAnalyzer(history_size=10)
    sample_requests = ["classification", "classification", "regression", "classification", "segmentation", "regression",
                       "classification", "classification", "segmentation", "classification"]

    for req in sample_requests:
        analyzer.record_request(req)

    print(f"Pattern Analysis: {analyzer.analyze_patterns()}")
    