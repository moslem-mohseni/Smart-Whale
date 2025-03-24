from typing import Dict, Any


class StrategySelector:
    """
    این کلاس مسئول انتخاب استراتژی بهبود کیفیت بر اساس شرایط داده است.
    """

    def __init__(self):
        """
        مقداردهی اولیه با تعریف استراتژی‌های موجود.
        """
        self.strategies = {
            "increase_score": self._increase_score,
            "apply_filter": self._apply_filter
        }

    def select_strategy(self, data: Dict[str, Any]) -> str:
        """
        انتخاب استراتژی مناسب بر اساس ویژگی‌های داده.
        """
        if data.get("quality_score", 1.0) < 0.5:
            return "increase_score"
        return "apply_filter"

    def apply_strategy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        اعمال استراتژی انتخاب‌شده بر روی داده.
        """
        strategy = self.select_strategy(data)
        return self.strategies[strategy](data)

    def _increase_score(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        افزایش کیفیت داده با افزایش مقدار امتیاز کیفیت.
        """
        data["quality_score"] = min(data["quality_score"] + 0.2, 1.0)
        return data

    def _apply_filter(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        اعمال فیلتر و اصلاح داده‌ها در صورت نیاز.
        """
        data["filtered"] = True
        return data
