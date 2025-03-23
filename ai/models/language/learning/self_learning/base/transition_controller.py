"""
TransitionController Module
---------------------------
این فایل مسئول مدیریت تغییر فاز (transition) مدل در سیستم خودآموزی است.
این کلاس با استفاده از داده‌های ارزیابی (evaluation metrics) و تعاریف فازها، تصمیم می‌گیرد که آیا مدل باید از یک فاز به فاز بعدی انتقال یابد.
تصمیم‌گیری بر مبنای مقایسه مقادیر واقعی با آستانه‌های تعریف‌شده در فازهای مختلف (BEGINNER، INTERMEDIATE، ADVANCED) انجام می‌شود.
این نسخه نهایی و عملیاتی با بهترین مکانیسم‌های هوشمند و کارایی بالا پیاده‌سازی شده است.
"""

import asyncio
import logging
from datetime import datetime
from enum import property
from typing import Dict, Any, Optional

from .phase_definitions import LearningPhase, PhaseDefinitions
from .phase_detector import PhaseDetector
from .phase_parameter_provider import PhaseParameterProvider
from .base_component import BaseComponent


class TransitionController(BaseComponent):
    """
    کلاس TransitionController مسئول مدیریت تغییر فاز مدل در سیستم خودآموزی است.

    ویژگی‌ها:
      - بررسی و مقایسه متریک‌های ارزیابی مدل با آستانه‌های تعریف‌شده.
      - تصمیم‌گیری برای انتقال فاز و به‌روزرسانی پارامترهای مدل.
      - ثبت رویدادهای تغییر فاز جهت نظارت و پیگیری.
    """

    def __init__(self,
                 phase_definitions: PhaseDefinitions,
                 phase_detector: PhaseDetector,
                 parameter_provider: PhaseParameterProvider,
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(component_type="transition_controller", config=config)
        self.phase_definitions = phase_definitions
        self.phase_detector = phase_detector
        self.parameter_provider = parameter_provider
        self.current_phase: LearningPhase = LearningPhase.BEGINNER
        # می‌توان تنظیمات اضافی مانند آستانه‌های انتقال را از پیکربندی گرفت.
        self.transition_threshold: float = self.config.get("transition_threshold", 0.1)
        self.logger.info(f"[TransitionController] Initialized with starting phase {self.current_phase.value}")

    async def evaluate_transition(self, evaluation_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        بررسی وضعیت مدل و تشخیص اینکه آیا تغییر فاز مورد نیاز است یا خیر.

        Args:
            evaluation_metrics (Dict[str, Any]): متریک‌های ارزیابی مانند coverage و teacher_dependency.

        Returns:
            Dict[str, Any]: شامل:
                - current_phase: فاز فعلی.
                - detected_phase: فاز تشخیص داده‌شده بر مبنای متریک‌ها.
                - transition_needed: آیا تغییر فاز لازم است.
                - details: جزئیات تصمیم‌گیری.
        """
        detection_result = self.phase_detector.detect_phase(evaluation_metrics)
        detected_phase_str = detection_result.get("detected_phase", LearningPhase.BEGINNER.value)
        detected_phase = LearningPhase(detected_phase_str)

        # تصمیم‌گیری: اگر فاز تشخیص داده‌شده با فاز فعلی تفاوت داشته باشد، انتقال مورد نیاز است.
        transition_needed = detected_phase != self.current_phase
        details = detection_result.get("details", {})

        self.logger.info(f"[TransitionController] Evaluation result: current_phase={self.current_phase.value}, "
                         f"detected_phase={detected_phase.value}, transition_needed={transition_needed}")
        return {
            "current_phase": self.current_phase.value,
            "detected_phase": detected_phase.value,
            "transition_needed": transition_needed,
            "details": details
        }

    async def perform_transition(self, evaluation_metrics: Dict[str, Any]) -> bool:
        """
        بررسی و انجام انتقال فاز در صورت لزوم.

        Args:
            evaluation_metrics (Dict[str, Any]): متریک‌های ارزیابی مدل.

        Returns:
            bool: True در صورت انتقال موفق، False در غیر این صورت.
        """
        decision = await self.evaluate_transition(evaluation_metrics)
        if decision.get("transition_needed"):
            new_phase = LearningPhase(decision["detected_phase"])
            self.logger.info(
                f"[TransitionController] Transitioning from {self.current_phase.value} to {new_phase.value}.")
            self.current_phase = new_phase
            # به‌روزرسانی پارامترهای مدل با استفاده از PhaseParameterProvider
            new_params = self.parameter_provider.get_parameters_for_phase(self.current_phase)
            self.logger.info(
                f"[TransitionController] Updated parameters for phase {self.current_phase.value}: {new_params}")
            # ثبت رویداد تغییر فاز
            await self.trigger_event("MODEL_PHASE_CHANGED", {
                "new_phase": self.current_phase.value,
                "updated_parameters": new_params,
                "timestamp": datetime.utcnow().isoformat()
            })
            return True
        else:
            self.logger.info(
                f"[TransitionController] No transition required. Model remains in {self.current_phase.value}.")
            return False

    def get_current_phase(self) -> LearningPhase:
        """
        دریافت فاز فعلی مدل.

        Returns:
            str: نام فاز فعلی.
        """
        return self.current_phase.value


# برای تست و توسعه:
if __name__ == "__main__":
    async def main():
        # فرض کنید ConfigManager در دسترس است و فایل پیکربندی مناسب را لود کرده‌ایم.
        from .config_manager import ConfigManager
        config_manager = ConfigManager(config_file="configs/learning/self_learning.yaml", load_immediately=True)
        phase_defs = PhaseDefinitions(config_manager)
        phase_detector = PhaseDetector(phase_defs)
        parameter_provider = PhaseParameterProvider(phase_defs)

        tc = TransitionController(phase_defs, phase_detector, parameter_provider, config={"transition_threshold": 0.1})
        # شبیه‌سازی متریک‌های ارزیابی
        sample_metrics = {"coverage": 0.75, "teacher_dependency": 0.3}
        transitioned = await tc.perform_transition(sample_metrics)
        print("Transition performed:", transitioned)
        print("Current phase:", tc.get_current_phase())


    asyncio.run(main())
