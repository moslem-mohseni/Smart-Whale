"""
PhaseDefinitions Module
-------------------------
این فایل شامل تعریف فازهای مختلف یادگیری (BEGINNER, INTERMEDIATE, ADVANCED) و پارامترهای مرتبط با هر فاز است.
تعاریف فازها از طریق فایل پیکربندی (self_learning.yaml) بارگذاری شده و امکان دسترسی و به‌روزرسانی تنظیمات هر فاز فراهم می‌شود.
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any

from .config_manager import ConfigManager  # فرض می‌شود فایل config_manager.py در همین پوشه موجود است.


class LearningPhase(Enum):
    BEGINNER = "BEGINNER"
    INTERMEDIATE = "INTERMEDIATE"
    ADVANCED = "ADVANCED"


@dataclass
class PhaseDefinition:
    teacher_dependency: float = field(default=0.0)
    coverage_threshold: float = field(default=0.0)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'PhaseDefinition':
        return PhaseDefinition(
            teacher_dependency=float(data.get("teacher_dependency", 0.0)),
            coverage_threshold=float(data.get("coverage_threshold", 0.0))
        )


class PhaseDefinitions:
    """
    کلاس PhaseDefinitions مسئول بارگذاری و نگهداری تعاریف فازهای یادگیری از پیکربندی سیستم است.
    """

    def __init__(self, config_manager: ConfigManager):
        self.logger = logging.getLogger("PhaseDefinitions")
        self.config_manager = config_manager
        self.phase_definitions: Dict[LearningPhase, PhaseDefinition] = {}
        self.load_definitions()

    def load_definitions(self) -> None:
        """
        بارگذاری تعاریف فازها از پیکربندی.
        """
        self.logger.info("[PhaseDefinitions] Loading phase definitions from configuration.")
        config = self.config_manager.get("self_learning.phases", {})
        for phase in LearningPhase:
            phase_data = config.get(phase.value)
            if phase_data:
                self.phase_definitions[phase] = PhaseDefinition.from_dict(phase_data)
                self.logger.debug(f"[PhaseDefinitions] Loaded {phase.value}: {self.phase_definitions[phase]}")
            else:
                self.logger.error(
                    f"[PhaseDefinitions] Configuration for phase '{phase.value}' not found. Using default values.")

    def get_phase_definition(self, phase: LearningPhase) -> PhaseDefinition:
        """
        دریافت تعریف یک فاز یادگیری.

        Args:
            phase (LearningPhase): فاز مورد نظر.

        Returns:
            PhaseDefinition: تعریف فاز با پارامترهای مربوطه.
        """
        return self.phase_definitions.get(phase, PhaseDefinition())

    def update_phase_definition(self, phase: LearningPhase, new_definition: Dict[str, Any]) -> None:
        """
        به‌روزرسانی تعریف یک فاز در حافظه.

        Args:
            phase (LearningPhase): فاز مورد نظر.
            new_definition (Dict[str, Any]): دیکشنری شامل مقادیر جدید برای به‌روزرسانی.
        """
        self.phase_definitions[phase] = PhaseDefinition.from_dict(new_definition)
        self.logger.info(f"[PhaseDefinitions] Updated definition for {phase.value}: {self.phase_definitions[phase]}")
