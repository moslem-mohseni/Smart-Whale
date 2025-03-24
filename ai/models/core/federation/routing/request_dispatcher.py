from typing import Dict, Any, Optional
from .route_optimizer import RouteOptimizer
from .load_balancer import LoadBalancer
from .priority_handler import PriorityHandler

class RequestDispatcher:
    """
    ماژول مدیریت و توزیع درخواست‌های فدراسیونی بین مدل‌های پردازشی.
    """

    def __init__(self):
        """
        مقداردهی اولیه و تنظیم ماژول‌های هوشمند برای پردازش درخواست‌ها.
        """
        self.route_optimizer = RouteOptimizer()
        self.load_balancer = LoadBalancer()
        self.priority_handler = PriorityHandler()

    def dispatch_request(self, request: Dict[str, Any]) -> Optional[str]:
        """
        توزیع درخواست به مدل مناسب بر اساس تحلیل مسیر، بار پردازشی و اولویت.
        :param request: دیکشنری شامل اطلاعات درخواست.
        :return: شناسه مدل انتخاب‌شده برای پردازش یا `None` در صورت عدم تخصیص.
        """
        if not isinstance(request, dict) or "data" not in request:
            raise ValueError("درخواست نامعتبر است! ساختار داده صحیح را ارسال کنید.")

        # بررسی و تخصیص اولویت به درخواست
        request_priority = self.priority_handler.evaluate_priority(request)

        # بهینه‌سازی مسیرهای پردازشی
        optimal_route = self.route_optimizer.optimize_route(request)

        # انتخاب مدل مناسب با توجه به بار کاری و مسیر بهینه‌شده
        selected_model = self.load_balancer.select_model(request, optimal_route, request_priority)

        if selected_model:
            # ارسال درخواست به مدل انتخاب‌شده
            return self._send_to_model(selected_model, request)

        return None  # عدم موفقیت در تخصیص مدل

    def _send_to_model(self, model: str, request: Dict[str, Any]) -> str:
        """
        ارسال درخواست به مدل انتخاب‌شده.
        :param model: شناسه مدل انتخاب‌شده.
        :param request: اطلاعات درخواست.
        :return: پیام تأیید ارسال موفق درخواست.
        """
        # شبیه‌سازی ارسال درخواست به مدل (در نسخه واقعی باید API مدل‌ها را فراخوانی کند)
        print(f"درخواست {request['data']} به مدل {model} ارسال شد.")
        return f"درخواست به مدل {model} ارسال شد."
