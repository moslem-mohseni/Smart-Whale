from typing import List, Dict, Any


class QualityOptimizer:
    """
    این کلاس مسئول بهینه‌سازی کیفیت داده‌ها و پیشنهاد راهکارهای بهبود است.
    """

    def __init__(self, optimization_factor: float = 0.1):
        """
        مقداردهی اولیه با ضریب بهینه‌سازی کیفیت داده‌ها.
        """
        self.optimization_factor = optimization_factor

    def optimize_data(self, data_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        اعمال بهینه‌سازی بر روی کیفیت داده‌ها.
        """
        optimized_data = []
        for data in data_batch:
            if "quality_score" in data:
                data["quality_score"] = min(data["quality_score"] + self.optimization_factor, 1.0)
                data["optimized"] = True
                optimized_data.append(data)
        return optimized_data
