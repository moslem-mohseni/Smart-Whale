from typing import Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class QualityEvaluator:
    """
    ماژول ارزیابی کیفیت خروجی مدل‌های فدراسیونی شامل تحلیل معنایی، انسجام، و سازگاری زمینه‌ای.
    """

    def __init__(self):
        """
        مقداردهی اولیه شامل تنظیم وزن‌های متریک‌های ارزیابی کیفیت.
        """
        self.weights = {
            "semantic": 0.4,  # وزن ارزیابی معنایی
            "coherence": 0.3,  # وزن بررسی انسجام
            "contextual": 0.3  # وزن ارزیابی تطبیق زمینه
        }

    def evaluate_response(self, response: str, context: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, float]:
        """
        ارزیابی جامع کیفیت پاسخ مدل.
        :param response: متن پاسخ تولیدشده توسط مدل.
        :param context: داده‌های زمینه‌ای شامل تاریخچه مکالمه و دانش قبلی مدل.
        :param requirements: الزامات کیفیت پاسخ شامل دقت مورد انتظار.
        :return: دیکشنری شامل امتیازات کیفیت پاسخ.
        """
        semantic_score = self._evaluate_semantic_accuracy(response, context["reference"])
        coherence_score = self._check_coherence(response, context["previous_responses"])
        context_score = self._validate_context_alignment(response, context["conversation_history"])

        final_score = self._calculate_weighted_score([
            (semantic_score, self.weights["semantic"]),
            (coherence_score, self.weights["coherence"]),
            (context_score, self.weights["contextual"])
        ])

        return {
            "semantic_score": semantic_score,
            "coherence_score": coherence_score,
            "context_score": context_score,
            "final_score": final_score
        }

    def _evaluate_semantic_accuracy(self, response: str, reference: str) -> float:
        """
        ارزیابی دقت معنایی پاسخ با استفاده از مقایسه شباهت برداری.
        :param response: پاسخ مدل.
        :param reference: پاسخ مرجع.
        :return: نمره شباهت معنایی.
        """
        response_vector = self._encode_text(response)
        reference_vector = self._encode_text(reference)

        return cosine_similarity([response_vector], [reference_vector])[0][0]

    def _check_coherence(self, response: str, previous_responses: list) -> float:
        """
        بررسی انسجام پاسخ با تحلیل شباهت معنایی آن نسبت به پاسخ‌های قبلی.
        :param response: پاسخ جدید مدل.
        :param previous_responses: لیستی از پاسخ‌های قبلی در مکالمه.
        :return: امتیاز انسجام.
        """
        if not previous_responses:
            return 1.0  # در صورت نبود پاسخ قبلی، انسجام در نظر گرفته نمی‌شود.

        similarities = [
            self._evaluate_semantic_accuracy(response, prev) for prev in previous_responses
        ]
        return np.mean(similarities)

    def _validate_context_alignment(self, response: str, conversation_history: list) -> float:
        """
        بررسی سازگاری پاسخ با زمینه مکالمه.
        :param response: پاسخ مدل.
        :param conversation_history: لیستی از پیام‌های قبلی در مکالمه.
        :return: نمره سازگاری زمینه‌ای.
        """
        if not conversation_history:
            return 1.0  # در صورت نبود زمینه قبلی، نمره حداکثر در نظر گرفته می‌شود.

        similarities = [
            self._evaluate_semantic_accuracy(response, msg) for msg in conversation_history
        ]
        return np.mean(similarities)

    def _encode_text(self, text: str) -> np.ndarray:
        """
        تبدیل متن به بردار عددی برای مقایسه شباهت معنایی.
        :param text: متن ورودی.
        :return: بردار عددی نمایانگر متن.
        """
        return np.array([ord(c) for c in text.lower()])[:300]  # تبدیل کاراکترها به مقادیر عددی محدود

    def _calculate_weighted_score(self, scores: list) -> float:
        """
        محاسبه میانگین وزنی امتیازات کیفیت پاسخ.
        :param scores: لیستی از جفت‌های (امتیاز، وزن).
        :return: امتیاز نهایی کیفیت.
        """
        weighted_sum = sum(score * weight for score, weight in scores)
        return weighted_sum / sum(weight for _, weight in scores)
