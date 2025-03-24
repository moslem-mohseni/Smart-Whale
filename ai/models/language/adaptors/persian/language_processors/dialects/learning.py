# language_processors/dialects/learning.py

"""
ماژول learning.py

این ماژول شامل کلاس DialectLearningProcessor است که وظیفه فراخوانی فرآیند یادگیری از خروجی مدل معلم (teacher output)
را بر عهده دارد. در اینجا متن ورودی ابتدا نرمال‌سازی می‌شود و سپس با استفاده از smart_model (مدل دانش‌آموز) یا در صورت لزوم
از teacher (مدل معلم) فرآیند یادگیری اجرا می‌شود. همچنین آمار مربوط به رویدادهای یادگیری ثبت و در صورت نیاز در پایگاه داده ذخیره می‌شود.
این ماژول به صورت مستقل عمل می‌کند و وابستگی به Kafka ندارد.
"""

import json
import logging
import time
from typing import Any, Dict, Optional, Union

import torch
import numpy as np

from ...config import CONFIG
from .data_access import DialectDataAccess
from ..utils.text_normalization import TextNormalizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DialectLearningProcessor:
    """
    DialectLearningProcessor مسئول فرآیند یادگیری از خروجی مدل معلم برای بهبود مدل دانش‌آموز در حوزه لهجه‌بندی است.

    ویژگی‌های کلیدی:
      - نرمال‌سازی متن ورودی.
      - فراخوانی متد learn_from_teacher در smart_model؛ در صورت عدم موفقیت، از teacher استفاده می‌شود.
      - ثبت و گزارش آمار رویدادهای یادگیری.
      - ذخیره رویدادهای یادگیری در پایگاه داده (در صورت نیاز به آنالیزهای بعدی).
    """

    def __init__(self, smart_model: Optional[Any] = None, teacher: Optional[Any] = None):
        self.logger = logger
        self.data_access = DialectDataAccess()  # استفاده از دسترسی متمرکز به داده‌ها و پایگاه داده
        self.normalizer = TextNormalizer()
        # بارگذاری مدل‌ها؛ در صورت ارائه مدل از پارامترها استفاده می‌کنیم
        self.smart_model = smart_model if smart_model is not None else self._load_smart_model()
        self.teacher = teacher if teacher is not None else self._load_teacher_model()
        self.learning_stats = {
            "teacher_learnings": 0,
            "smart_model_updates": 0,
            "total_learning_events": 0,
            "last_learning_time": None
        }
        self.logger.info("DialectLearningProcessor initialized.")

    def _load_smart_model(self) -> Optional[Any]:
        try:
            module = __import__(f"ai.models.language.adaptors.{self.data_access.language}.smart_model",
                                fromlist=["SmartModel"])
            return module.SmartModel()
        except Exception as e:
            self.logger.error(f"خطا در بارگذاری مدل هوشمند برای یادگیری: {e}")
            return None

    def _load_teacher_model(self) -> Optional[Any]:
        try:
            module = __import__(f"ai.models.language.adaptors.{self.data_access.language}.teacher",
                                fromlist=["TeacherModel"])
            return module.TeacherModel()
        except Exception as e:
            self.logger.warning("مدل معلم یافت نشد؛ فرآیند یادگیری محدود خواهد بود.")
            return None

    def learn_from_teacher(self, text: Union[str, torch.Tensor], teacher_output: Any) -> bool:
        """
        فراخوانی فرآیند یادگیری از خروجی مدل معلم برای به‌روزرسانی مدل دانش‌آموز.

        Args:
            text (Union[str, torch.Tensor]): متن ورودی جهت یادگیری.
            teacher_output (Any): خروجی مدل معلم (شامل اطلاعات آموزشی برای بهبود مدل).

        Returns:
            bool: نتیجه یادگیری (True در صورت موفقیت، False در غیر این صورت).
        """
        # نرمال‌سازی متن
        if isinstance(text, torch.Tensor):
            # در صورتیکه متن به صورت تنسور باشد، آن را به رشته تبدیل می‌کنیم
            text_str = text.item() if hasattr(text, "item") else str(text)
        else:
            text_str = text
        normalized_text = self.normalizer.normalize(text_str)
        self.logger.info("شروع فرآیند یادگیری از معلم برای متن: %s", normalized_text[:50])

        # تلاش اولیه از مدل دانش‌آموز (smart_model)
        if self.smart_model:
            try:
                update_result = self.smart_model.learn_from_teacher(normalized_text, teacher_output)
                self.learning_stats["smart_model_updates"] += 1
                self.learning_stats["total_learning_events"] += 1
                self.learning_stats["last_learning_time"] = time.time()
                self.logger.info("یادگیری از معلم با موفقیت در مدل دانش‌آموز انجام شد.")
                self._save_learning_event(normalized_text, teacher_output, True)
                return update_result
            except Exception as e:
                self.logger.error(f"خطا در یادگیری از مدل دانش‌آموز: {e}")

        # در صورت عدم موفقیت، تلاش از مدل معلم
        if self.teacher:
            try:
                teacher_learn_result = self.teacher.learn_from_teacher(normalized_text, teacher_output)
                self.learning_stats["teacher_learnings"] += 1
                self.learning_stats["total_learning_events"] += 1
                self.learning_stats["last_learning_time"] = time.time()
                self.logger.info("یادگیری از معلم با موفقیت در مدل معلم انجام شد.")
                self._save_learning_event(normalized_text, teacher_output, True)
                return teacher_learn_result
            except Exception as e:
                self.logger.error(f"خطا در یادگیری از مدل معلم: {e}")

        self.logger.warning("هیچ مدلی برای فرآیند یادگیری موجود نیست؛ یادگیری انجام نشد.")
        self._save_learning_event(normalized_text, teacher_output, False)
        return False

    def _save_learning_event(self, text: str, teacher_output: Any, success: bool) -> None:
        """
        ذخیره رویداد یادگیری در پایگاه داده برای آنالیزهای بعدی.

        Args:
            text (str): متن ورودی.
            teacher_output (Any): خروجی مدل معلم.
            success (bool): نتیجه موفقیت‌آمیز بودن فرآیند یادگیری.
        """
        event = {
            "text": text,
            "teacher_output": teacher_output,
            "learning_success": success,
            "timestamp": time.time()
        }
        try:
            # در صورت وجود جدول "dialect_learning_history" در پایگاه داده، رویداد ثبت می‌شود
            self.data_access.database.insert_data("dialect_learning_history", event)
            self.logger.info("رویداد یادگیری با موفقیت در پایگاه داده ثبت شد.")
        except Exception as e:
            self.logger.error(f"خطا در ذخیره رویداد یادگیری: {e}")

    def get_learning_statistics(self) -> Dict[str, Any]:
        """
        دریافت آمار و اطلاعات مربوط به رویدادهای یادگیری.

        Returns:
            Dict[str, Any]: شامل تعداد به‌روزرسانی‌های مدل دانش‌آموز، رویدادهای مدل معلم،
                           تعداد کل رویدادها و زمان آخرین به‌روزرسانی.
        """
        return self.learning_stats


if __name__ == "__main__":
    # مثال کاربردی از DialectLearningProcessor
    processor = DialectLearningProcessor()
    sample_text = "این یک متن نمونه برای یادگیری است."
    # فرض کنید teacher_output خروجی مدل معلم باشد (برای نمونه، یک دیکشنری ساده)
    sample_teacher_output = {"info": "خروجی نمونه معلم جهت بهبود مدل"}
    success = processor.learn_from_teacher(sample_text, sample_teacher_output)
    print("نتیجه یادگیری:", success)
    print("آمار یادگیری:", processor.get_learning_statistics())
