import numpy as np
from core.monitoring.metrics.collector import MetricsCollector

class QualityChecker:
    """
    ماژولی برای بررسی کیفیت داده‌های پردازشی.
    این ماژول داده‌های پرت، ناسازگار و بی‌کیفیت را شناسایی کرده و پیشنهادهای بهبود ارائه می‌دهد.
    """

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.outlier_threshold = 3  # آستانه تشخیص داده‌های پرت (۳ انحراف معیار)
        self.missing_data_threshold = 0.1  # اگر بیش از ۱۰٪ داده‌ها گم شده باشند، هشدار بدهد

    async def evaluate_data_quality(self, data_stream: list) -> dict:
        """
        بررسی کیفیت داده‌های پردازشی و شناسایی مشکلات.

        :param data_stream: لیستی از داده‌های ورودی
        :return: دیکشنری شامل تحلیل کیفیت داده‌ها
        """
        if not data_stream:
            return {"status": "no_data", "message": "هیچ داده‌ای برای تحلیل وجود ندارد."}

        # بررسی داده‌های پرت
        outliers = self._detect_outliers(data_stream)

        # بررسی میزان داده‌های ناقص
        missing_data_ratio = self._calculate_missing_data_ratio(data_stream)

        # تحلیل کیفیت کلی داده‌ها
        quality_report = {
            "total_data_points": len(data_stream),
            "outliers_detected": len(outliers),
            "missing_data_ratio": missing_data_ratio,
            "data_consistency": self._evaluate_consistency(data_stream),
            "quality_score": self._calculate_quality_score(outliers, missing_data_ratio)
        }

        return quality_report

    def _detect_outliers(self, data_stream: list) -> list:
        """
        شناسایی داده‌های پرت (Outliers) با استفاده از تحلیل آماری.

        :param data_stream: لیست داده‌ها
        :return: لیستی از داده‌های پرت
        """
        data_array = np.array(data_stream)
        mean = np.mean(data_array)
        std_dev = np.std(data_array)

        outliers = [x for x in data_array if abs(x - mean) > self.outlier_threshold * std_dev]
        return outliers

    def _calculate_missing_data_ratio(self, data_stream: list) -> float:
        """
        محاسبه درصد داده‌های گم‌شده یا نامعتبر.

        :param data_stream: لیست داده‌ها
        :return: نسبت داده‌های نامعتبر
        """
        total = len(data_stream)
        missing_count = sum(1 for x in data_stream if x is None or x == "" or np.isnan(x))
        return missing_count / total if total > 0 else 0

    def _evaluate_consistency(self, data_stream: list) -> float:
        """
        بررسی یکنواختی داده‌ها برای تشخیص داده‌های ناسازگار.

        :param data_stream: لیست داده‌ها
        :return: عددی بین 0 تا 1 که میزان یکنواختی داده‌ها را نشان می‌دهد.
        """
        unique_values = len(set(data_stream))
        return 1 - (unique_values / len(data_stream)) if len(data_stream) > 0 else 0

    def _calculate_quality_score(self, outliers: list, missing_data_ratio: float) -> float:
        """
        محاسبه امتیاز کیفیت داده‌ها.

        :param outliers: لیست داده‌های پرت
        :param missing_data_ratio: نسبت داده‌های نامعتبر
        :return: امتیاز کیفیت داده‌ها بین ۰ تا ۱
        """
        penalty_outliers = min(len(outliers) / 100, 0.5)  # حداکثر 50٪ نمره کسر شود
        penalty_missing = min(missing_data_ratio * 2, 0.5)  # حداکثر 50٪ نمره کسر شود

        return max(1 - (penalty_outliers + penalty_missing), 0)

