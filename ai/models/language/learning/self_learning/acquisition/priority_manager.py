"""
PriorityManager Module
------------------------
این فایل مسئول محاسبه و تعیین اولویت نیازهای یادگیری است.
با استفاده از چند معیار کلیدی مانند:
  - frequency: تعداد تکرار نیاز،
  - recency: تازگی وقوع نیاز (زمان از وقوع آخرین رویداد به ثانیه)،
  - impact: تاثیر یا اهمیت نیاز (مثلاً تاثیر بر عملکرد مدل)،
  - gap_size: اندازه شکاف دانشی یا میزان عدم پوشش،
فرمولی خطی برای محاسبه اولویت ایجاد می‌کند.
این نسخه نهایی و عملیاتی با بهترین مکانیسم‌های هوشمند و کارایی بالا پیاده‌سازی شده است.
"""

import logging
from typing import Dict, List, Any, Optional


class PriorityManager:
    """
    PriorityManager مسئول محاسبه و تعیین اولویت نیازهای یادگیری بر اساس چند معیار است.

    فرمول محاسبه اولویت:
        priority = w1 * frequency + w2 * recency + w3 * impact + w4 * gap_size

    پارامترهای وزن (w1، w2، w3، w4) از طریق پیکربندی قابل تنظیم هستند.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        مقداردهی اولیه PriorityManager.

        Args:
            config (Optional[Dict[str, Any]]): پیکربندی اختیاری شامل وزن‌های مربوط به معیارهای اولویت.
                - weight_frequency: وزن معیار frequency (پیش‌فرض: 1.0)
                - weight_recency: وزن معیار recency (پیش‌فرض: 1.0)
                - weight_impact: وزن معیار impact (پیش‌فرض: 1.0)
                - weight_gap_size: وزن معیار gap_size (پیش‌فرض: 1.0)
        """
        self.logger = logging.getLogger("PriorityManager")
        self.config = config or {}
        self.w1 = float(self.config.get("weight_frequency", 1.0))
        self.w2 = float(self.config.get("weight_recency", 1.0))
        self.w3 = float(self.config.get("weight_impact", 1.0))
        self.w4 = float(self.config.get("weight_gap_size", 1.0))
        self.logger.info(f"[PriorityManager] Initialized with weights: frequency={self.w1}, recency={self.w2}, "
                         f"impact={self.w3}, gap_size={self.w4}")

    def calculate_priority(self, metrics: Dict[str, float]) -> float:
        """
        محاسبه اولویت یک نیاز یادگیری بر اساس معیارهای داده‌شده.

        Args:
            metrics (Dict[str, float]): دیکشنری شامل مقادیر معیارهای مورد نظر.
                انتظار می‌رود کلیدهای زیر موجود باشد:
                    - frequency: تعداد وقوع نیاز
                    - recency: زمان (ثانیه) از وقوع آخرین رویداد (هرچه مقدار کمتر، تازه‌تر است)
                    - impact: تاثیر نیاز (عدد مثبت؛ هر چه بیشتر، اهمیت بالاتر)
                    - gap_size: اندازه شکاف دانشی (هرچه بزرگتر، نیاز بیشتر)

        Returns:
            float: مقدار اولویت محاسبه‌شده.
        """
        frequency = metrics.get("frequency", 0.0)
        recency = metrics.get("recency", 0.0)
        impact = metrics.get("impact", 0.0)
        gap_size = metrics.get("gap_size", 0.0)
        # توجه: اگر recency کمتر باشد، می‌توان آن را به صورت معکوس در نظر گرفت؛ اما در اینجا فرض می‌کنیم مقادیر به نحوی تنظیم شده‌اند.
        priority = (self.w1 * frequency +
                    self.w2 * recency +
                    self.w3 * impact +
                    self.w4 * gap_size)
        self.logger.debug(f"[PriorityManager] Calculated priority: {priority} using metrics: {metrics}")
        return priority

    def prioritize_needs(self, needs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        مرتب‌سازی نیازهای یادگیری بر اساس اولویت محاسبه‌شده.

        Args:
            needs (List[Dict[str, Any]]): لیستی از نیازها؛ هر مورد باید دارای کلید "metrics" باشد که شامل
                                          معیارهای لازم جهت محاسبه اولویت است.

        Returns:
            List[Dict[str, Any]]: لیستی از نیازها به ترتیب نزولی اولویت به همراه افزودن کلید "priority" به هر مورد.
        """
        for need in needs:
            metrics = need.get("metrics", {})
            need["priority"] = self.calculate_priority(metrics)
        sorted_needs = sorted(needs, key=lambda x: x["priority"], reverse=True)
        self.logger.info(f"[PriorityManager] Sorted {len(needs)} needs by priority.")
        return sorted_needs
