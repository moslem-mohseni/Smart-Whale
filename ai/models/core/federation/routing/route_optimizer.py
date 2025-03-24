from typing import Dict, Any, List
from collections import defaultdict

class RouteOptimizer:
    """
    ماژول بهینه‌سازی مسیرهای پردازش درخواست‌ها در فدراسیون.
    """

    def __init__(self):
        """
        مقداردهی اولیه با نگهداری اطلاعات مسیرهای پردازشی و مدل‌های بهینه‌ی مرتبط.
        """
        self.route_data: Dict[str, List[str]] = defaultdict(list)  # مسیر -> مدل‌های مناسب
        self.route_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)  # مسیر -> {مدل: زمان پاسخ}

    def optimize_route(self, request: Dict[str, Any]) -> str:
        """
        بهینه‌سازی مسیر پردازشی بر اساس داده‌های مسیرهای گذشته.
        :param request: اطلاعات درخواست شامل نوع پردازش و نیازهای منابع.
        :return: مسیر بهینه برای هدایت درخواست.
        """
        request_type = request.get("type", "default")

        if request_type in self.route_data and self.route_data[request_type]:
            # انتخاب بهترین مسیر بر اساس کمترین تأخیر پردازشی
            optimal_route = min(self.route_metrics[request_type], key=self.route_metrics[request_type].get)
            return optimal_route

        return "default_route"  # در صورت نبود اطلاعات کافی، مسیر پیش‌فرض انتخاب می‌شود

    def update_route_performance(self, route: str, model: str, response_time: float):
        """
        بروزرسانی داده‌های مسیر بر اساس زمان پاسخ مدل‌ها.
        :param route: مسیر مربوطه.
        :param model: مدل پردازشی که درخواست را انجام داده است.
        :param response_time: زمان پاسخ مدل (ms).
        """
        if model not in self.route_data[route]:
            self.route_data[route].append(model)

        self.route_metrics[route][model] = response_time

    def get_optimized_models(self, route: str) -> List[str]:
        """
        دریافت فهرست مدل‌های بهینه برای یک مسیر خاص.
        :param route: مسیر موردنظر.
        :return: لیست مدل‌های پردازشی مناسب.
        """
        return sorted(self.route_metrics[route], key=self.route_metrics[route].get) if route in self.route_metrics else []

    def get_route_status(self) -> Dict[str, Any]:
        """
        دریافت وضعیت مسیرها و مدل‌های مرتبط با هر مسیر.
        :return: دیکشنری شامل اطلاعات مسیرها و مدل‌های بهینه‌شده.
        """
        return {
            "route_data": self.route_data,
            "route_metrics": self.route_metrics
        }
