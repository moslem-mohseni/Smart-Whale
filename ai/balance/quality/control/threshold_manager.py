from typing import Dict, Any


class ThresholdManager:
    """
    این کلاس مسئول مدیریت آستانه‌های کیفیت داده‌ها است.
    """

    def __init__(self, min_quality: float = 0.75):
        """
        مقداردهی اولیه با حداقل میزان کیفیت قابل قبول.
        """
        self.min_quality = min_quality

    def is_valid(self, data: Dict[str, Any]) -> bool:
        """
        بررسی می‌کند که آیا کیفیت داده از حداقل میزان مجاز فراتر رفته است یا خیر.
        """
        return data.get("quality_score", 1.0) >= self.min_quality

    def adjust_threshold(self, new_threshold: float) -> None:
        """
        تنظیم حداقل میزان کیفیت مورد نیاز برای اعتبار داده‌ها.
        """
        self.min_quality = max(0.0, min(new_threshold, 1.0))
