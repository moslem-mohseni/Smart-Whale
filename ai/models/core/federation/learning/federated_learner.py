from typing import Dict, Any, List, Optional
from .model_aggregator import ModelAggregator
from .learning_optimizer import LearningOptimizer
from ..knowledge_sharing.privacy_guard import PrivacyGuard

class FederatedLearner:
    """
    ماژول مدیریت یادگیری فدراسیونی در بین مدل‌های هوش مصنوعی.
    """

    def __init__(self):
        """
        مقداردهی اولیه سیستم یادگیری فدراسیونی.
        """
        self.model_aggregator = ModelAggregator()
        self.learning_optimizer = LearningOptimizer()
        self.privacy_guard = PrivacyGuard()
        self.model_updates: Dict[str, List[float]] = {}  # ذخیره به‌روزرسانی‌های مدل‌ها

    def collect_model_update(self, model_id: str, updates: List[float]) -> None:
        """
        جمع‌آوری به‌روزرسانی‌های وزن‌های مدل از مدل‌های فدراسیونی.
        :param model_id: شناسه مدل ارسال‌کننده به‌روزرسانی.
        :param updates: لیست وزن‌های جدید مدل.
        """
        protected_updates = self.privacy_guard.protect_privacy({"updates": updates})
        self.model_updates[model_id] = protected_updates["updates"]

    def aggregate_and_update_models(self) -> Optional[List[float]]:
        """
        تجمیع به‌روزرسانی‌های دریافت‌شده از مدل‌ها و ایجاد یک نسخه‌ی جدید از مدل بهینه.
        :return: وزن‌های جدید مدل ترکیب‌شده یا `None` اگر داده کافی نباشد.
        """
        if not self.model_updates:
            return None  # اگر هیچ مدلی به‌روزرسانی ارسال نکرده باشد

        aggregated_weights = self.model_aggregator.aggregate_updates(self.model_updates)

        # پاک‌سازی به‌روزرسانی‌های قبلی
        self.model_updates.clear()

        return aggregated_weights

    def optimize_learning_parameters(self, loss_history: List[float]) -> float:
        """
        بهینه‌سازی پارامترهای یادگیری بر اساس تاریخچه‌ی کاهش خطا.
        :param loss_history: لیست کاهش خطای مدل در مراحل قبلی یادگیری.
        :return: نرخ یادگیری بهینه‌شده.
        """
        return self.learning_optimizer.adjust_learning_rate(loss_history)

    def get_collected_updates(self) -> Dict[str, List[float]]:
        """
        دریافت لیست به‌روزرسانی‌های جمع‌آوری‌شده از مدل‌های فدراسیونی.
        :return: دیکشنری شامل به‌روزرسانی‌های مدل‌ها.
        """
        return self.model_updates
