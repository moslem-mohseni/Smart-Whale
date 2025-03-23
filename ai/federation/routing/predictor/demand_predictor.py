from typing import List, Dict
from collections import deque


class DemandPredictor:
    """
    پیش‌بینی میزان درخواست‌های آینده و نیاز پردازشی برای بهینه‌سازی تخصیص منابع
    """

    def __init__(self, history_size: int = 10):
        self.request_history = deque(maxlen=history_size)  # نگهداری تاریخچه آخرین درخواست‌ها

    def record_request(self, request_type: str) -> None:
        """
        ثبت درخواست در تاریخچه برای تحلیل روند
        :param request_type: نوع درخواست (مانند classification, regression)
        """
        self.request_history.append(request_type)

    def predict_demand(self) -> Dict[str, int]:
        """
        تحلیل تاریخچه درخواست‌ها و پیش‌بینی میزان تقاضا برای انواع درخواست‌ها
        :return: دیکشنری شامل پیش‌بینی میزان درخواست برای هر نوع درخواست
        """
        demand_forecast = {}
        for req in self.request_history:
            demand_forecast[req] = demand_forecast.get(req, 0) + 1
        return demand_forecast


# نمونه استفاده از DemandPredictor برای تست
if __name__ == "__main__":
    predictor = DemandPredictor(history_size=5)
    sample_requests = ["classification", "regression", "classification", "segmentation", "classification"]

    for req in sample_requests:
        predictor.record_request(req)

    print(f"Predicted Demand: {predictor.predict_demand()}")
    