from typing import List, Dict
from collections import Counter


class PathAnalyzer:
    """
    تحلیلگر مسیر برای بررسی روند استفاده از مسیرهای پردازشی و ارائه پیشنهادات بهینه
    """

    def __init__(self):
        self.path_history = []  # تاریخچه مسیرهای انتخاب‌شده

    def record_path(self, path: str) -> None:
        """
        ثبت مسیر پردازشی استفاده‌شده در تاریخچه
        :param path: نام مسیر
        """
        self.path_history.append(path)

    def get_most_frequent_paths(self, top_n: int = 3) -> List[str]:
        """
        دریافت پرتکرارترین مسیرهای استفاده‌شده برای تحلیل بهینه‌سازی
        :param top_n: تعداد مسیرهای پرتکرار مورد نظر
        :return: لیست مسیرهای پرتکرار
        """
        counter = Counter(self.path_history)
        return [path for path, _ in counter.most_common(top_n)]

    def analyze_path_efficiency(self) -> Dict[str, int]:
        """
        تحلیل مسیرهای استفاده‌شده و میزان استفاده از آن‌ها
        :return: دیکشنری شامل مسیرها و تعداد دفعات استفاده‌شده
        """
        return dict(Counter(self.path_history))


# نمونه استفاده از PathAnalyzer برای تست
if __name__ == "__main__":
    analyzer = PathAnalyzer()
    sample_paths = ["route_a", "route_b", "route_a", "route_c", "route_b", "route_a"]

    for path in sample_paths:
        analyzer.record_path(path)

    print(f"Most Frequent Paths: {analyzer.get_most_frequent_paths()}")
    print(f"Path Usage Analysis: {analyzer.analyze_path_efficiency()}")
    