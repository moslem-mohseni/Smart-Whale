from typing import Dict, Any, List
from collections import defaultdict

class RequirementAnalyzer:
    """
    تحلیل‌گر نیازهای داده‌ای برای مدل‌های مختلف در سیستم Balance.
    این ماژول داده‌های مورد نیاز هر مدل را بررسی کرده و وضعیت تعادل داده‌ای را ارزیابی می‌کند.
    """

    def __init__(self):
        # ذخیره میزان استفاده از داده‌ها برای هر مدل
        self.data_usage = defaultdict(lambda: {"requests": 0, "usage": 0})

    def log_data_request(self, model_id: str, data_size: int) -> None:
        """
        ثبت درخواست‌های مدل برای داده و به‌روزرسانی میزان مصرف آن.

        :param model_id: شناسه مدل
        :param data_size: میزان داده درخواست‌شده (برحسب بایت)
        """
        self.data_usage[model_id]["requests"] += 1
        self.data_usage[model_id]["usage"] += data_size

    def analyze_needs(self, model_id: str, recent_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        تحلیل نیازهای داده‌ای یک مدل بر اساس درخواست‌های اخیر.

        :param model_id: شناسه مدل
        :param recent_requests: لیست درخواست‌های اخیر مدل
        :return: دیکشنری شامل تحلیل نیازها و میزان مصرف
        """
        if not recent_requests:
            return {
                "model_id": model_id,
                "status": "No recent requests",
                "data_usage": self.data_usage[model_id]
            }

        total_size = sum(req["data_size"] for req in recent_requests)
        average_size = total_size / len(recent_requests)

        return {
            "model_id": model_id,
            "total_requests": len(recent_requests),
            "total_data_requested": total_size,
            "average_data_size": average_size,
            "data_usage": self.data_usage[model_id]
        }

    def detect_data_shortage(self, model_id: str, data_threshold: int) -> bool:
        """
        بررسی کمبود داده برای یک مدل خاص.

        :param model_id: شناسه مدل
        :param data_threshold: حد آستانه برای کمبود داده (برحسب بایت)
        :return: `True` اگر داده کافی نباشد، در غیر اینصورت `False`
        """
        return self.data_usage[model_id]["usage"] < data_threshold
