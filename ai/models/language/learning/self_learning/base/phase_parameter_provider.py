"""
PhaseParameterProvider Module
-------------------------------
این فایل مسئول فراهم آوردن پارامترهای تنظیم‌شده متناسب با هر فاز یادگیری است.
با استفاده از تعاریف فازها و پیکربندی‌های مربوطه، این کلاس پارامترهایی مانند نرخ یادگیری، اندازه دسته (batch size) و سایر تنظیمات مهم را برای هر فاز ارائه می‌دهد.
"""

import logging
from typing import Dict, Any

from .phase_definitions import LearningPhase, PhaseDefinitions


class PhaseParameterProvider:
    """
    کلاس PhaseParameterProvider پارامترهای لازم برای تنظیم مدل در هر فاز یادگیری (BEGINNER, INTERMEDIATE, ADVANCED) را فراهم می‌کند.

    ویژگی‌ها:
      - استفاده از PhaseDefinitions جهت دریافت مقادیر پایه مانند teacher_dependency و coverage_threshold.
      - محاسبه پارامترهای دینامیک مانند نرخ یادگیری (learning rate)، اندازه دسته (batch size)، و تعداد تکرار (max_iterations) متناسب با هر فاز.
      - امکان به‌روزرسانی پیکربندی در صورت نیاز.
    """

    def __init__(self, phase_definitions: PhaseDefinitions, config: Dict[str, Any] = None):
        self.logger = logging.getLogger("PhaseParameterProvider")
        self.phase_definitions = phase_definitions
        # پیکربندی اضافی (اختیاری) می‌تواند شامل تنظیمات پیش‌فرض برای پارامترها باشد.
        self.config = config or {}
        self.logger.info("[PhaseParameterProvider] Initialized with provided phase definitions.")

    def get_parameters_for_phase(self, phase: LearningPhase) -> Dict[str, Any]:
        """
        دریافت پارامترهای تنظیم‌شده برای فاز یادگیری مشخص.

        Args:
            phase (LearningPhase): فاز مورد نظر.

        Returns:
            Dict[str, Any]: دیکشنری شامل پارامترهایی مانند:
                - learning_rate: نرخ یادگیری متناسب با فاز.
                - batch_size: اندازه دسته مورد استفاده در آموزش.
                - teacher_dependency: مقدار وابستگی به مدل معلم از تنظیمات فاز.
                - coverage_threshold: مقدار پوشش دانشی مورد انتظار.
                - max_iterations: حداکثر تکرارهای آموزش در این فاز.
        """
        # دریافت تعریف فاز از PhaseDefinitions
        phase_def = self.phase_definitions.get_phase_definition(phase)
        self.logger.debug(f"[PhaseParameterProvider] Retrieved definition for {phase.value}: {phase_def}")

        # تنظیم پارامترها بر اساس فاز؛ این مقادیر می‌توانند از پیکربندی خارجی یا الگوریتم‌های بهینه‌سازی دینامیک نیز تأمین شوند.
        if phase == LearningPhase.BEGINNER:
            params = {
                "learning_rate": 0.01,
                "batch_size": 32,
                "teacher_dependency": phase_def.teacher_dependency,
                "coverage_threshold": phase_def.coverage_threshold,
                "max_iterations": 1000
            }
        elif phase == LearningPhase.INTERMEDIATE:
            params = {
                "learning_rate": 0.005,
                "batch_size": 64,
                "teacher_dependency": phase_def.teacher_dependency,
                "coverage_threshold": phase_def.coverage_threshold,
                "max_iterations": 2000
            }
        elif phase == LearningPhase.ADVANCED:
            params = {
                "learning_rate": 0.001,
                "batch_size": 128,
                "teacher_dependency": phase_def.teacher_dependency,
                "coverage_threshold": phase_def.coverage_threshold,
                "max_iterations": 3000
            }
        else:
            self.logger.warning(f"[PhaseParameterProvider] Unknown phase: {phase}. Using default parameters.")
            params = {
                "learning_rate": 0.01,
                "batch_size": 32,
                "teacher_dependency": 0.9,
                "coverage_threshold": 0.3,
                "max_iterations": 1000
            }
        self.logger.info(f"[PhaseParameterProvider] Parameters for {phase.value}: {params}")
        return params

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        به‌روزرسانی پیکربندی داخلی جهت تغییر پارامترهای عمومی.

        Args:
            new_config (Dict[str, Any]): پیکربندی جدید جهت به‌روزرسانی.
        """
        self.config.update(new_config)
        self.logger.info("[PhaseParameterProvider] Configuration updated.")
