"""
PhaseDetector Module
----------------------
این فایل مسئول تشخیص فاز فعلی مدل بر اساس معیارهای عملکرد و وضعیت فعلی است.
با استفاده از تعاریف فازها (PhaseDefinitions) که از طریق ConfigManager بارگذاری می‌شود،
این کلاس تعیین می‌کند که مدل در کدام فاز (BEGINNER, INTERMEDIATE, ADVANCED) قرار دارد.
این تصمیم‌گیری می‌تواند بر پایه معیارهایی مانند پوشش دانشی (coverage) و وابستگی به معلم (teacher_dependency)
یا سایر متریک‌های کلیدی باشد.
"""

import logging
from typing import Dict, Any

from .phase_definitions import LearningPhase, PhaseDefinitions, PhaseDefinition
from ai.core.monitoring.metrics.collector import MetricsCollector


class PhaseDetector:
    """
    کلاس PhaseDetector برای تشخیص فاز فعلی مدل.

    ویژگی‌ها:
      - استفاده از PhaseDefinitions جهت دریافت آستانه‌ها و پارامترهای هر فاز.
      - دریافت ورودی‌های کلیدی (مثلاً متریک‌های ارزیابی) جهت تصمیم‌گیری.
      - ارائه‌ی فاز تشخیص داده‌شده به همراه جزئیات.
    """

    def __init__(self, phase_definitions: PhaseDefinitions, metrics: MetricsCollector = None):
        self.logger = logging.getLogger("PhaseDetector")
        self.phase_definitions = phase_definitions
        self.metrics = metrics  # می‌توان متریک‌های عملکردی مدل را برای تشخیص فاز به کار برد
        self.logger.info("[PhaseDetector] Initialized with provided phase definitions.")

    def detect_phase(self, evaluation_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        تشخیص فاز فعلی مدل بر اساس متریک‌های ارزیابی ورودی.

        Args:
            evaluation_metrics (Dict[str, Any]): دیکشنری حاوی متریک‌های کلیدی مانند:
                - coverage: میزان پوشش دانشی مدل (مقدار بین 0 تا 1)
                - teacher_dependency: میزان وابستگی مدل به معلم (مقدار بین 0 تا 1)
                - سایر متریک‌های مورد نیاز

        Returns:
            Dict[str, Any]: دیکشنری شامل:
                - detected_phase: فاز تشخیص داده‌شده (LearningPhase)
                - details: جزئیات تصمیم‌گیری به صورت دیکشنری (مانند اختلاف مقادیر واقعی با آستانه)
        """
        coverage = evaluation_metrics.get("coverage", 0.0)
        teacher_dependency = evaluation_metrics.get("teacher_dependency", 1.0)
        self.logger.debug(
            f"[PhaseDetector] Received evaluation metrics: coverage={coverage}, teacher_dependency={teacher_dependency}")

        # دریافت تعاریف فازها
        beginner_def: PhaseDefinition = self.phase_definitions.get_phase_definition(LearningPhase.BEGINNER)
        intermediate_def: PhaseDefinition = self.phase_definitions.get_phase_definition(LearningPhase.INTERMEDIATE)
        advanced_def: PhaseDefinition = self.phase_definitions.get_phase_definition(LearningPhase.ADVANCED)

        details = {}
        detected_phase = LearningPhase.BEGINNER  # پیش‌فرض

        # استراتژی تشخیص فاز:
        # اگر پوشش مدل کمتر از آستانه فاز BEGINNER است و وابستگی به معلم بالا است، فاز BEGINNER.
        # اگر پوشش بین آستانه BEGINNER و INTERMEDIATE باشد و وابستگی کاهش یافته باشد، فاز INTERMEDIATE.
        # در غیر این صورت، اگر پوشش بالاتر از آستانه ADVANCED است و وابستگی بسیار پایین است، فاز ADVANCED.
        if coverage < beginner_def.coverage_threshold or teacher_dependency >= beginner_def.teacher_dependency:
            detected_phase = LearningPhase.BEGINNER
            details = {
                "reason": "Low coverage or high teacher dependency",
                "coverage": coverage,
                "expected_coverage": beginner_def.coverage_threshold,
                "teacher_dependency": teacher_dependency,
                "expected_teacher_dependency": beginner_def.teacher_dependency
            }
        elif coverage < intermediate_def.coverage_threshold or teacher_dependency > intermediate_def.teacher_dependency:
            detected_phase = LearningPhase.INTERMEDIATE
            details = {
                "reason": "Moderate coverage and moderate teacher dependency",
                "coverage": coverage,
                "expected_coverage": intermediate_def.coverage_threshold,
                "teacher_dependency": teacher_dependency,
                "expected_teacher_dependency": intermediate_def.teacher_dependency
            }
        else:
            detected_phase = LearningPhase.ADVANCED
            details = {
                "reason": "High coverage and low teacher dependency",
                "coverage": coverage,
                "expected_coverage": advanced_def.coverage_threshold,
                "teacher_dependency": teacher_dependency,
                "expected_teacher_dependency": advanced_def.teacher_dependency
            }

        self.logger.info(f"[PhaseDetector] Detected phase: {detected_phase.value} with details: {details}")
        return {"detected_phase": detected_phase.value, "details": details}



