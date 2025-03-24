from typing import List, Dict, Any


class FeedbackAnalyzer:
    """
    این کلاس مسئول تحلیل بازخوردهای کیفیتی داده‌ها و ارائه پیشنهادات بهبود است.
    """

    def __init__(self):
        """
        مقداردهی اولیه.
        """
        self.feedback_history = []

    def collect_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        جمع‌آوری بازخوردهای کیفیتی داده‌ها.
        """
        self.feedback_history.append(feedback)

    def analyze_feedback(self) -> Dict[str, float]:
        """
        تحلیل بازخوردهای دریافتی و ارائه خلاصه‌ای از وضعیت کیفیت داده‌ها.
        """
        if not self.feedback_history:
            return {"average_score": 1.0, "issues_detected": 0}

        avg_score = sum(fb["quality_score"] for fb in self.feedback_history) / len(self.feedback_history)
        issues_count = sum(1 for fb in self.feedback_history if fb.get("issue_detected", False))

        return {
            "average_score": round(avg_score, 2),
            "issues_detected": issues_count
        }

    def reset_feedback(self) -> None:
        """
        پاک‌سازی بازخوردهای ذخیره‌شده.
        """
        self.feedback_history.clear()
