# فایل نهایی config_manager.py برای مدیریت پیکربندی Self-Learning
# ----------------------------------------------------------------------------

"""
مدیریت پیکربندی سیستم خودآموزی (ConfigManager)

این ماژول وظیفه دارد پیکربندی ماژول Self-Learning را از فایل `self_learning.yaml` لود کرده،
اعتبارسنجی کند و توابع کمکی برای دسترسی به مقادیر آن ارائه دهد.
"""

import yaml
import logging
import os
from typing import Any, Dict, Optional, List
from pathlib import Path


class ConfigManager:
    """
    مدیریت پیکربندی سیستم خودآموزی (Self-Learning) از فایل `self_learning.yaml`
    """

    def __init__(self, config_file: str = "configs/learning/self_learning.yaml", load_immediately: bool = True):
        """
        راه‌اندازی مدیریت‌کننده پیکربندی

        Args:
            config_file: مسیر فایل پیکربندی
            load_immediately: اگر True باشد، بلافاصله بارگذاری و اعتبارسنجی انجام می‌گیرد
        """
        self.logger = logging.getLogger("SelfLearningConfigManager")

        # مسیر فایل پیکربندی
        self.config_path = Path(config_file)

        # دیکشنری اصلی تنظیمات
        self.config: Dict[str, Any] = {}
        self.validation_errors: List[str] = []

        if load_immediately:
            self.load_config()

        self.logger.info("ConfigManager initialized")

    def load_config(self) -> bool:
        """
        بارگذاری تنظیمات از فایل پیکربندی `self_learning.yaml`
        Returns:
            bool: نتیجه بارگذاری موفقیت‌آمیز یا عدم موفقیت
        """
        try:
            if not self.config_path.exists():
                self.logger.error(f"[ConfigManager] Config file not found: {self.config_path}")
                return False

            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)

            self.logger.info(f"[ConfigManager] Configuration loaded from {self.config_path}")

            # اعتبارسنجی تنظیمات
            return self.validate_config()

        except Exception as e:
            self.logger.error(f"[ConfigManager] Error loading config: {str(e)}")
            return False

    def validate_config(self) -> bool:
        """
        اعتبارسنجی تنظیمات بارگذاری‌شده
        Returns:
            bool: نتیجه اعتبارسنجی
        """
        self.validation_errors.clear()

        # کلیدهای ضروری که باید وجود داشته باشد
        required_keys = [
            "self_learning",
            "self_learning.base",
            "self_learning.phases"
        ]

        for key in required_keys:
            if not self.has_key(key):
                self.validation_errors.append(f"Required configuration key not found: {key}")

        # بررسی وجود فازهای BEGINNER, INTERMEDIATE, ADVANCED
        phases = self.get("self_learning.phases", {})
        if not phases or not isinstance(phases, dict):
            self.validation_errors.append("self_learning.phases must be a dictionary")
        else:
            needed_phases = ["BEGINNER", "INTERMEDIATE", "ADVANCED"]
            for phase in needed_phases:
                if phase not in phases:
                    self.validation_errors.append(f"Missing phase: {phase}")

        if self.validation_errors:
            for error in self.validation_errors:
                self.logger.error(f"[ConfigManager] Validation error: {error}")
            return False

        self.logger.info("[ConfigManager] Configuration validated successfully")
        return True

    def has_key(self, key: str) -> bool:
        """
        بررسی وجود یک کلید در پیکربندی، بر اساس مسیر نقطه‌گذاری
        Args:
            key: مسیر کلید به فرم "self_learning.base"
        Returns:
            bool: وجود یا عدم وجود کلید
        """
        keys = key.split('.')
        current = self.config
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return False
        return True

    def get(self, key: str, default: Any = None) -> Any:
        """
        دریافت مقدار یک کلید از پیکربندی
        Args:
            key: مسیر کلید با جداکننده نقطه
            default: مقدار پیش‌فرض در صورت عدم وجود کلید
        Returns:
            Any: مقدار کلید یا مقدار پیش‌فرض
        """
        keys = key.split('.')
        current = self.config
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

    def get_validation_errors(self) -> List[str]:
        """
        دریافت خطاهای اعتبارسنجی پیکربندی
        Returns:
            List[str]: لیست خطاها
        """
        return list(self.validation_errors)

    def reload(self) -> bool:
        """
        بارگذاری مجدد تنظیمات از فایل پیکربندی
        Returns:
            bool: نتیجه عملیات
        """
        return self.load_config()

    def get_self_learning_config(self) -> Dict[str, Any]:
        """
        دریافت تمام تنظیمات مربوط به self_learning
        Returns:
            Dict[str, Any]: تنظیمات self_learning
        """
        return self.get("self_learning", {})

    def get_base_config(self) -> Dict[str, Any]:
        """
        دریافت تنظیمات بخش "base" در self_learning
        Returns:
            Dict[str, Any]: دیکشنری تنظیمات پایه
        """
        return self.get("self_learning.base", {})

    def get_phase_config(self, phase: str) -> Dict[str, Any]:
        """
        دریافت تنظیمات مربوط به یک فاز خاص
        Args:
            phase: نام فاز (BEGINNER, INTERMEDIATE, ADVANCED)
        Returns:
            Dict[str, Any]: تنظیمات آن فاز
        """
        return self.get(f"self_learning.phases.{phase}", {})

    def get_phase_dependency(self, phase: str) -> float:
        """
        میانگین وابستگی به معلم (teacher_dependency) یک فاز
        """
        phase_config = self.get_phase_config(phase)
        return float(phase_config.get("teacher_dependency", 0.0))

    def get_phase_coverage_threshold(self, phase: str) -> float:
        """
        مقدار پوشش دانشی (coverage_threshold) برای فاز
        """
        phase_config = self.get_phase_config(phase)
        return float(phase_config.get("coverage_threshold", 0.0))
