from typing import List, Dict


class PreloadManager:
    """
    مدیریت پیش‌بارگذاری مدل‌ها برای کاهش تأخیر در پاسخ‌دهی و افزایش سرعت پردازش
    """

    def __init__(self, max_preload: int = 3):
        self.preloaded_models = []  # لیست مدل‌های پیش‌بارگذاری‌شده
        self.max_preload = max_preload

    def preload_model(self, model_name: str) -> None:
        """
        پیش‌بارگذاری یک مدل در حافظه برای پردازش سریع‌تر
        :param model_name: نام مدل موردنظر برای پیش‌بارگذاری
        """
        if model_name not in self.preloaded_models:
            if len(self.preloaded_models) >= self.max_preload:
                self.preloaded_models.pop(0)  # حذف قدیمی‌ترین مدل در صورت پر شدن ظرفیت
            self.preloaded_models.append(model_name)

    def get_preloaded_models(self) -> List[str]:
        """
        دریافت لیست مدل‌های پیش‌بارگذاری‌شده
        :return: لیستی از نام مدل‌های موجود در حافظه
        """
        return self.preloaded_models

    def is_model_preloaded(self, model_name: str) -> bool:
        """
        بررسی اینکه آیا مدل مشخصی از قبل در حافظه پیش‌بارگذاری شده است یا خیر
        :param model_name: نام مدل مورد بررسی
        :return: مقدار بولین که نشان می‌دهد مدل پیش‌بارگذاری شده است یا نه
        """
        return model_name in self.preloaded_models


# نمونه استفاده از PreloadManager برای تست
if __name__ == "__main__":
    manager = PreloadManager(max_preload=2)
    manager.preload_model("model_a")
    manager.preload_model("model_b")
    manager.preload_model("model_c")  # باید مدل_a را حذف کند چون ظرفیت پر شده است

    print(f"Preloaded Models: {manager.get_preloaded_models()}")
    print(f"Is 'model_a' preloaded? {manager.is_model_preloaded('model_a')}")
    print(f"Is 'model_b' preloaded? {manager.is_model_preloaded('model_b')}")

