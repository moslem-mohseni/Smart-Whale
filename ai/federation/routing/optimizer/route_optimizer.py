from typing import Dict, Any, List


class RouteOptimizer:
    """
    بهینه‌ساز مسیر برای انتخاب بهترین مسیر پردازشی در ماژول Federation
    """

    def __init__(self):
        self.route_history = []  # نگهداری تاریخچه مسیرهای پردازشی

    def optimize_route(self, request: Dict[str, Any], available_routes: List[str]) -> str:
        """
        انتخاب بهینه‌ترین مسیر برای پردازش درخواست بر اساس تاریخچه و اولویت‌بندی
        :param request: اطلاعات درخواست پردازشی
        :param available_routes: لیست مسیرهای موجود برای پردازش
        :return: بهترین مسیر انتخاب‌شده
        """
        if not available_routes:
            raise ValueError("No available routes for optimization.")

        best_route = min(available_routes, key=self._route_efficiency_score)
        self.route_history.append(best_route)
        return best_route

    def _route_efficiency_score(self, route: str) -> float:
        """
        محاسبه امتیاز کارایی مسیر برای بهینه‌سازی
        :param route: نام مسیر
        :return: امتیاز مسیر (کمتر بهتر)
        """
        recent_usage = self.route_history.count(route)
        return recent_usage  # هر چه مسیر کمتر استفاده شده باشد، اولویت بیشتری دارد

    def get_route_history(self) -> List[str]:
        """
        دریافت تاریخچه مسیرهای انتخاب‌شده برای بررسی و تحلیل
        :return: لیست مسیرهای گذشته
        """
        return self.route_history


# نمونه استفاده از RouteOptimizer برای تست
if __name__ == "__main__":
    optimizer = RouteOptimizer()
    routes = ["route_a", "route_b", "route_c"]
    request = {"type": "classification"}

    best_route = optimizer.optimize_route(request, routes)
    print(f"Optimized Route: {best_route}")
    print(f"Route History: {optimizer.get_route_history()}")
    