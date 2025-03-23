from typing import Dict, Any


class CostCalculator:
    """
    محاسبه هزینه پردازشی مسیرهای مختلف برای انتخاب بهینه‌ترین مسیر
    """

    def __init__(self):
        self.cost_data = {
            "route_a": 1.2,
            "route_b": 0.8,
            "route_c": 1.5
        }  # داده‌های اولیه هزینه مسیرها

    def calculate_cost(self, route: str, request: Dict[str, Any]) -> float:
        """
        محاسبه هزینه مسیر بر اساس اطلاعات درخواست و مسیر انتخاب‌شده
        :param route: نام مسیر مورد بررسی
        :param request: اطلاعات درخواست پردازشی
        :return: هزینه پردازشی مسیر
        """
        base_cost = self.cost_data.get(route, 1.0)  # دریافت هزینه پایه مسیر
        complexity_factor = request.get("complexity", 1.0)  # ضریب پیچیدگی درخواست
        return base_cost * complexity_factor

    def get_cheapest_route(self, routes: Dict[str, Dict[str, Any]]) -> str:
        """
        انتخاب ارزان‌ترین مسیر پردازشی از میان مسیرهای موجود
        :param routes: دیکشنری شامل مسیرها و درخواست‌های مرتبط با هر مسیر
        :return: نام مسیر با کمترین هزینه پردازشی
        """
        if not routes:
            raise ValueError("No routes provided for cost comparison.")

        return min(routes.keys(), key=lambda route: self.calculate_cost(route, routes[route]))


# نمونه استفاده از CostCalculator برای تست
if __name__ == "__main__":
    calculator = CostCalculator()
    test_request = {"complexity": 1.2}
    test_routes = {
        "route_a": test_request,
        "route_b": {"complexity": 1.1},
        "route_c": {"complexity": 1.3}
    }

    print(f"Cost of route_a: {calculator.calculate_cost('route_a', test_request)}")
    print(f"Cheapest Route: {calculator.get_cheapest_route(test_routes)}")
