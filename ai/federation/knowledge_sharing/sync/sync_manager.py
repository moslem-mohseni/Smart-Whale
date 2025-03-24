from typing import Dict, Any


class SyncManager:
    """
    مدیریت همگام‌سازی دانش بین مدل‌ها برای حفظ یکپارچگی داده‌ها
    """

    def __init__(self):
        self.sync_state: Dict[str, Any] = {}  # وضعیت همگام‌سازی مدل‌ها

    def update_sync_state(self, model_name: str, version: str) -> None:
        """
        به‌روزرسانی وضعیت همگام‌سازی مدل‌ها
        :param model_name: نام مدل
        :param version: نسخه جدید دانش مدل
        """
        self.sync_state[model_name] = version

    def get_sync_state(self, model_name: str) -> str:
        """
        دریافت نسخه فعلی دانش یک مدل
        :param model_name: نام مدل
        :return: نسخه ذخیره‌شده یا مقدار پیش‌فرض "unknown"
        """
        return self.sync_state.get(model_name, "unknown")

    def is_synced(self, model_name: str, version: str) -> bool:
        """
        بررسی اینکه آیا مدل موردنظر با نسخه خاصی همگام است
        :param model_name: نام مدل
        :param version: نسخه مورد بررسی
        :return: مقدار بولین که نشان‌دهنده همگام بودن یا نبودن مدل است
        """
        return self.sync_state.get(model_name) == version


# نمونه استفاده از SyncManager برای تست
if __name__ == "__main__":
    sync_manager = SyncManager()
    sync_manager.update_sync_state("model_a", "v1.2")
    print(f"Sync State for model_a: {sync_manager.get_sync_state('model_a')}")
    print(f"Is model_a synced with v1.2? {sync_manager.is_synced('model_a', 'v1.2')}")
    print(f"Is model_a synced with v1.3? {sync_manager.is_synced('model_a', 'v1.3')}")
