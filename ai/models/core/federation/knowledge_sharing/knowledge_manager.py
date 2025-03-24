from typing import Dict, Any
from collections import defaultdict
from .privacy_guard import PrivacyGuard

class KnowledgeManager:
    """
    ماژول مدیریت دانش بین مدل‌های فدراسیونی.
    """

    def __init__(self):
        """
        مقداردهی اولیه سیستم مدیریت دانش.
        """
        self.knowledge_base: Dict[str, Any] = defaultdict(dict)  # مدل -> دانش مرتبط
        self.privacy_guard = PrivacyGuard()

    def store_knowledge(self, model_id: str, knowledge: Dict[str, Any]) -> None:
        """
        ذخیره دانش برای یک مدل خاص.
        :param model_id: شناسه مدل.
        :param knowledge: داده‌های دانش برای ذخیره‌سازی.
        """
        protected_knowledge = self.privacy_guard.protect_privacy(knowledge)
        self.knowledge_base[model_id] = protected_knowledge

    def retrieve_knowledge(self, model_id: str) -> Dict[str, Any]:
        """
        دریافت دانش ذخیره‌شده برای یک مدل خاص.
        :param model_id: شناسه مدل.
        :return: دانش مدل.
        """
        return self.knowledge_base.get(model_id, {})

    def update_knowledge(self, model_id: str, new_knowledge: Dict[str, Any]) -> None:
        """
        به‌روزرسانی دانش یک مدل خاص.
        :param model_id: شناسه مدل.
        :param new_knowledge: داده‌های جدید دانش.
        """
        protected_knowledge = self.privacy_guard.protect_privacy(new_knowledge)
        self.knowledge_base[model_id].update(protected_knowledge)

    def delete_knowledge(self, model_id: str) -> None:
        """
        حذف دانش یک مدل خاص.
        :param model_id: شناسه مدل.
        """
        if model_id in self.knowledge_base:
            del self.knowledge_base[model_id]

    def get_all_knowledge(self) -> Dict[str, Dict[str, Any]]:
        """
        دریافت تمام دانش‌های ذخیره‌شده در فدراسیون.
        :return: دیکشنری شامل دانش تمام مدل‌ها.
        """
        return self.knowledge_base
