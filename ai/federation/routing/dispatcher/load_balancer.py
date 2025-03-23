import asyncio
from typing import List, Dict, Any, Optional


class LoadBalancer:
    """
    مدیریت و تخصیص درخواست‌ها به مدل‌های موجود با استفاده از الگوریتم متعادل‌سازی بار
    """

    def __init__(self):
        self.models = ["model_a", "model_b", "model_c"]  # لیست مدل‌های موجود
        self.model_loads = {model: 0 for model in self.models}  # بار پردازشی هر مدل

    async def select_model(self, request: Dict[str, Any]) -> Optional[str]:
        """
        انتخاب یک مدل برای پردازش درخواست بر اساس میزان بار پردازشی کمترین مدل
        :param request: درخواست ورودی شامل اطلاعات مربوط به پردازش
        :return: نام مدل انتخاب‌شده یا None در صورت عدم وجود مدل آماده
        """
        available_models = sorted(self.model_loads.items(), key=lambda x: x[1])

        for model, load in available_models:
            if load < self.get_max_load_threshold():
                self.model_loads[model] += 1  # افزایش بار پردازشی مدل انتخاب‌شده
                return model

        return None  # اگر هیچ مدلی ظرفیت پردازش نداشته باشد

    async def release_model(self, model: str) -> None:
        """
        کاهش بار پردازشی یک مدل پس از پردازش درخواست
        :param model: نام مدل پردازشی
        """
        if model in self.model_loads and self.model_loads[model] > 0:
            self.model_loads[model] -= 1

    def get_max_load_threshold(self) -> int:
        """
        دریافت حداکثر بار پردازشی مجاز برای هر مدل
        :return: مقدار آستانه بار پردازشی
        """
        return 5  # مقدار پیش‌فرض آستانه بار مدل‌ها


# نمونه استفاده از LoadBalancer برای تست
if __name__ == "__main__":
    async def test_load_balancer():
        lb = LoadBalancer()
        for _ in range(10):
            model = await lb.select_model({"type": "classification"})
            print(f"Selected Model: {model}")
            if model:
                await lb.release_model(model)


    asyncio.run(test_load_balancer())