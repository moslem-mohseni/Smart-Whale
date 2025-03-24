from typing import List, Dict, Any
from .threshold_manager import ThresholdManager
from .action_executor import ActionExecutor


class QualityController:
    """
    این کلاس مسئول کنترل کیفیت داده‌ها و اعمال اقدامات اصلاحی در صورت نیاز است.
    """

    def __init__(self, threshold_manager: ThresholdManager, action_executor: ActionExecutor):
        """
        مقداردهی اولیه با مدیریت آستانه‌ها و اجرای اقدامات اصلاحی.
        """
        self.threshold_manager = threshold_manager
        self.action_executor = action_executor

    def validate_data_quality(self, data_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        بررسی کیفیت داده‌ها و مشخص کردن داده‌های خارج از استاندارد.
        """
        invalid_data = []
        for data in data_batch:
            if not self.threshold_manager.is_valid(data):
                invalid_data.append(data)
        return invalid_data

    def apply_corrections(self, invalid_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        اجرای اقدامات اصلاحی برای داده‌های نامعتبر.
        """
        corrected_data = []
        for data in invalid_data:
            corrected = self.action_executor.correct_data(data)
            if corrected:
                corrected_data.append(corrected)
        return corrected_data
