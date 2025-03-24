from typing import Dict, Any, List, Optional
from .route_optimizer import RouteOptimizer

class LoadBalancer:
    """
    ماژول متعادل‌سازی بار بین مدل‌های پردازشی برای جلوگیری از ازدحام منابع.
    """

    def __init__(self, strategy: str = "least_connections"):
        """
        مقداردهی اولیه متعادل‌کننده بار.
        :param strategy: استراتژی توزیع بار (`least_connections`, `round_robin`).
        """
        if strategy not in ["least_connections", "round_robin"]:
            raise ValueError("استراتژی انتخابی نامعتبر است. گزینه‌های معتبر: 'least_connections' یا 'round_robin'.")

        self.strategy = strategy
        self.model_connections: Dict[str, int] = {}  # تعداد پردازش‌های فعال در هر مدل
        self.models: List[str] = []  # لیست مدل‌های در دسترس
        self.route_optimizer = RouteOptimizer()
        self.last_assigned_index = 0  # برای Round Robin

    def register_model(self, model_id: str):
        """
        ثبت یک مدل جدید در سیستم توزیع بار.
        :param model_id: شناسه مدل.
        """
        if model_id not in self.models:
            self.models.append(model_id)
            self.model_connections[model_id] = 0

    def deregister_model(self, model_id: str):
        """
        حذف یک مدل از سیستم توزیع بار.
        :param model_id: شناسه مدل.
        """
        if model_id in self.models:
            self.models.remove(model_id)
            del self.model_connections[model_id]

    def select_model(self, request: Dict[str, Any], optimal_route: str, priority: int) -> Optional[str]:
        """
        انتخاب بهترین مدل برای پردازش درخواست بر اساس استراتژی تعریف‌شده.
        :param request: اطلاعات درخواست برای تحلیل بار کاری.
        :param optimal_route: مسیر بهینه محاسبه‌شده برای پردازش.
        :param priority: اولویت درخواست.
        :return: شناسه مدل انتخاب‌شده برای پردازش.
        """
        if not self.models:
            return None  # اگر هیچ مدلی در دسترس نباشد

        # تحلیل مسیر برای بررسی مدل‌های سازگار با مسیر بهینه
        optimized_models = self.route_optimizer.get_optimized_models(optimal_route)

        # فیلتر کردن مدل‌ها بر اساس مدل‌های سازگار با مسیر و مدل‌های ثبت‌شده
        available_models = [model for model in optimized_models if model in self.models]

        if not available_models:
            return None  # هیچ مدل مناسبی برای مسیر وجود ندارد

        if self.strategy == "least_connections":
            return min(available_models, key=lambda model: self.model_connections.get(model, 0))

        elif self.strategy == "round_robin":
            self.last_assigned_index = (self.last_assigned_index + 1) % len(available_models)
            return available_models[self.last_assigned_index]

    def release_model(self, model_id: str):
        """
        آزادسازی منابع مدل پس از اتمام پردازش.
        :param model_id: شناسه مدل.
        """
        if model_id in self.model_connections and self.model_connections[model_id] > 0:
            self.model_connections[model_id] -= 1

    def get_status(self) -> Dict[str, Any]:
        """
        دریافت وضعیت فعلی متعادل‌سازی بار.
        :return: دیکشنری شامل اطلاعات بار پردازشی مدل‌ها.
        """
        return {
            "strategy": self.strategy,
            "model_connections": self.model_connections
        }
