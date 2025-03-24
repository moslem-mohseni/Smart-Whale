from typing import Dict, Any


class ConflictResolver:
    """
    حل تعارض‌های دانش بین مدل‌ها برای حفظ یکپارچگی داده‌ها
    """

    def __init__(self):
        self.conflict_log: Dict[str, Any] = {}  # ثبت تعارض‌های کشف‌شده

    def detect_conflict(self, model_name: str, local_version: str, remote_version: str) -> bool:
        """
        بررسی اینکه آیا نسخه دانش محلی و نسخه دانش از راه دور دچار تعارض هستند یا نه
        :param model_name: نام مدل مورد بررسی
        :param local_version: نسخه ذخیره‌شده محلی
        :param remote_version: نسخه دانش دریافتی از راه دور
        :return: مقدار بولین که نشان می‌دهد تعارض وجود دارد یا نه
        """
        conflict_exists = local_version != remote_version
        if conflict_exists:
            self.conflict_log[model_name] = (local_version, remote_version)
        return conflict_exists

    def resolve_conflict(self, model_name: str, resolution_strategy: str = "latest") -> str:
        """
        حل تعارض بین نسخه‌های مختلف دانش بر اساس استراتژی مشخص
        :param model_name: نام مدل دارای تعارض
        :param resolution_strategy: استراتژی حل تعارض (latest، local، remote)
        :return: نسخه نهایی پس از حل تعارض
        """
        if model_name not in self.conflict_log:
            return "No conflict detected"

        local_version, remote_version = self.conflict_log[model_name]
        if resolution_strategy == "latest":
            resolved_version = max(local_version, remote_version)
        elif resolution_strategy == "local":
            resolved_version = local_version
        elif resolution_strategy == "remote":
            resolved_version = remote_version
        else:
            raise ValueError("Invalid resolution strategy")

        del self.conflict_log[model_name]  # حذف تعارض پس از حل آن
        return resolved_version


# نمونه استفاده از ConflictResolver برای تست
if __name__ == "__main__":
    resolver = ConflictResolver()
    has_conflict = resolver.detect_conflict("model_a", "v1.2", "v1.3")
    print(f"Conflict Detected: {has_conflict}")

    if has_conflict:
        resolved_version = resolver.resolve_conflict("model_a", "latest")
        print(f"Resolved Version: {resolved_version}")
