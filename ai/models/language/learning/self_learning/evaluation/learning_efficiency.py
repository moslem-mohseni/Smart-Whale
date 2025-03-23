"""
LearningEfficiency Module
---------------------------
این فایل مسئول ارزیابی کارایی فرآیند یادگیری مدل در سیستم خودآموزی است.
این کلاس، با استفاده از متریک‌های مربوط به دوره‌های آموزشی (مانند کاهش loss، افزایش دقت، زمان صرف‌شده و ...)
کارایی یادگیری را اندازه‌گیری می‌کند و شاخصی از بهبود عملکرد مدل به ازای هر واحد زمان را ارائه می‌دهد.
این نسخه نهایی و عملیاتی با بهترین مکانیسم‌های هوشمند و کارایی بالا پیاده‌سازی شده است.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from ..base.base_component import BaseComponent


class LearningEfficiency(BaseComponent):
    """
    LearningEfficiency مسئول ارزیابی و گزارش کارایی فرآیند یادگیری مدل است.

    ورودی مورد انتظار هر دوره آموزشی:
      {
         "loss_before": float,       # مقدار loss قبل از آموزش
         "loss_after": float,        # مقدار loss بعد از آموزش
         "cycle_duration": float     # مدت زمان اجرای دوره آموزشی به ثانیه
      }

    خروجی:
      {
         "improvement_rate": float,   # درصد بهبود loss به ازای هر ثانیه (مثلاً 0.002 به معنای کاهش 0.2% loss در هر ثانیه)
         "details": str               # توضیحات تکمیلی در مورد کارایی یادگیری
      }
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(component_type="learning_efficiency", config=config)
        self.logger = logging.getLogger("LearningEfficiency")
        # تنظیمات اختیاری می‌تواند شامل حداقل بهبود قابل قبول باشد
        self.min_improvement_rate = float(self.config.get("min_improvement_rate", 0.0))
        self.logger.info(f"[LearningEfficiency] Initialized with min_improvement_rate={self.min_improvement_rate}")

    def evaluate_efficiency(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        ارزیابی کارایی فرآیند یادگیری بر اساس تغییر در loss و زمان صرف‌شده.

        Args:
            metrics (Dict[str, float]): شامل:
                - loss_before: مقدار loss قبل از دوره آموزشی.
                - loss_after: مقدار loss بعد از دوره آموزشی.
                - cycle_duration: مدت زمان اجرای دوره آموزشی به ثانیه.

        Returns:
            Dict[str, Any]: شامل:
                - improvement_rate: بهبود loss به ازای هر ثانیه.
                - details: توضیحات در خصوص کارایی.
        """
        try:
            loss_before = metrics.get("loss_before")
            loss_after = metrics.get("loss_after")
            cycle_duration = metrics.get("cycle_duration")
            if loss_before is None or loss_after is None or cycle_duration is None or cycle_duration <= 0:
                raise ValueError("Missing or invalid metrics for learning efficiency evaluation.")

            # درصد بهبود loss در کل دوره (به صورت نسبی)
            overall_improvement = (loss_before - loss_after) / loss_before if loss_before > 0 else 0.0
            # بهبود به ازای هر ثانیه
            improvement_rate = overall_improvement / cycle_duration

            details = (f"Loss decreased from {loss_before:.4f} to {loss_after:.4f} "
                       f"in {cycle_duration:.2f} seconds, resulting in an improvement rate of {improvement_rate:.6f} per second.")
            self.logger.info(f"[LearningEfficiency] {details}")
            self.increment_metric("learning_efficiency_evaluated")
            return {
                "improvement_rate": round(improvement_rate, 6),
                "details": details
            }
        except Exception as e:
            self.logger.error(f"[LearningEfficiency] Error during efficiency evaluation: {str(e)}")
            self.record_error_metric()
            return {
                "improvement_rate": 0.0,
                "details": f"Error: {str(e)}"
            }


# Sample usage for testing (final version intended for production)
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)

    le = LearningEfficiency(config={"min_improvement_rate": 0.0001})

    # شبیه‌سازی داده‌های یک دوره آموزشی
    sample_metrics = {
        "loss_before": 0.50,
        "loss_after": 0.45,
        "cycle_duration": 100.0  # ثانیه
    }

    result = le.evaluate_efficiency(sample_metrics)
    print("Learning Efficiency Evaluation:")
    print(result)
