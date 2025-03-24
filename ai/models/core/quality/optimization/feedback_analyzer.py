from typing import Dict, Any, List
import numpy as np
from .quality_optimizer import QualityOptimizer


class FeedbackAnalyzer:
    """
    ماژول تحلیل بازخورد کاربران و استفاده از آن برای بهبود کیفیت مدل‌های فدراسیونی.
    """

    def __init__(self):
        """
        مقداردهی اولیه و تنظیم متغیرهای تحلیل بازخورد.
        """
        self.feedback_data: Dict[str, List[Dict[str, Any]]] = {}
        self.quality_optimizer = QualityOptimizer()

    def collect_feedback(self, model_id: str, feedback: Dict[str, Any]):
        """
        جمع‌آوری بازخورد کاربران درباره عملکرد مدل.
        :param model_id: شناسه مدل.
        :param feedback: دیکشنری شامل نظر کاربران شامل امتیازات و نظرات متنی.
        """
        if model_id not in self.feedback_data:
            self.feedback_data[model_id] = []

        self.feedback_data[model_id].append(feedback)

    def analyze_feedback(self, model_id: str) -> Dict[str, float]:
        """
        تحلیل بازخورد کاربران و تولید متریک‌های کیفیت.
        :param model_id: شناسه مدل.
        :return: دیکشنری شامل تحلیل کیفیت مدل بر اساس بازخورد کاربران.
        """
        if model_id not in self.feedback_data or len(self.feedback_data[model_id]) == 0:
            return {"error": "No feedback available for analysis."}

        scores = {
            "accuracy": [],
            "coherence": [],
            "contextual_fit": [],
            "user_satisfaction": []
        }

        for feedback in self.feedback_data[model_id]:
            scores["accuracy"].append(feedback.get("accuracy", 0))
            scores["coherence"].append(feedback.get("coherence", 0))
            scores["contextual_fit"].append(feedback.get("contextual_fit", 0))
            scores["user_satisfaction"].append(feedback.get("user_satisfaction", 0))

        avg_scores = {key: np.mean(values) for key, values in scores.items()}

        return avg_scores

    def improve_model_based_on_feedback(self, model_id: str):
        """
        بهینه‌سازی مدل بر اساس تحلیل بازخورد کاربران.
        :param model_id: شناسه مدل.
        """
        feedback_analysis = self.analyze_feedback(model_id)
        if "error" in feedback_analysis:
            return feedback_analysis  # اگر داده‌ای موجود نبود، بهینه‌سازی انجام نشود.

        # ارسال داده‌های تحلیل‌شده به QualityOptimizer برای تنظیمات مدل
        return self.quality_optimizer.optimize_model_quality(model_id, feedback_analysis)

    def get_feedback_history(self, model_id: str) -> List[Dict[str, Any]]:
        """
        دریافت تاریخچه‌ی بازخورد کاربران برای یک مدل خاص.
        :param model_id: شناسه مدل.
        :return: لیست بازخوردهای دریافت‌شده برای مدل.
        """
        return self.feedback_data.get(model_id, [])
