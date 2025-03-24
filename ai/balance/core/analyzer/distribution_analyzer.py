from typing import Dict, Any, List
from collections import defaultdict

class DistributionAnalyzer:
    """
    تحلیل‌گر توزیع داده‌ها بین مدل‌های مختلف برای بررسی تعادل مصرف داده.
    """

    def __init__(self):
        # ذخیره میزان توزیع داده برای هر مدل
        self.data_distribution = defaultdict(lambda: {"total_data": 0, "request_count": 0})

    def log_data_usage(self, model_id: str, data_size: int) -> None:
        """
        ثبت میزان مصرف داده برای یک مدل مشخص.

        :param model_id: شناسه مدل
        :param data_size: میزان داده مصرف‌شده (برحسب بایت)
        """
        self.data_distribution[model_id]["total_data"] += data_size
        self.data_distribution[model_id]["request_count"] += 1

    def analyze_distribution(self) -> Dict[str, Any]:
        """
        تحلیل توزیع داده‌ها بین مدل‌ها و شناسایی عدم تعادل در مصرف داده.

        :return: دیکشنری شامل میانگین مصرف داده و مدل‌های با مصرف نامتعادل
        """
        if not self.data_distribution:
            return {"status": "No data recorded"}

        total_data_used = sum(model["total_data"] for model in self.data_distribution.values())
        total_requests = sum(model["request_count"] for model in self.data_distribution.values())
        average_data_usage = total_data_used / len(self.data_distribution) if self.data_distribution else 0

        imbalance_models = {
            model_id: data for model_id, data in self.data_distribution.items()
            if data["total_data"] > 1.5 * average_data_usage or data["total_data"] < 0.5 * average_data_usage
        }

        return {
            "total_data_used": total_data_used,
            "total_requests": total_requests,
            "average_data_usage": average_data_usage,
            "imbalanced_models": imbalance_models
        }

    def detect_imbalance(self, threshold: float = 1.5) -> List[str]:
        """
        شناسایی مدل‌هایی که مصرف داده‌ی آن‌ها خارج از حد تعادل است.

        :param threshold: آستانه عدم تعادل (۱.۵ برابر میانگین مصرف)
        :return: لیست مدل‌هایی که مصرف داده‌ی آن‌ها نامتعادل است.
        """
        analysis = self.analyze_distribution()
        return list(analysis["imbalanced_models"].keys()) if "imbalanced_models" in analysis else []
