"""
FeedbackAnalyzer Module
-------------------------
این فایل مسئول تحلیل بازخوردهای کاربران درباره عملکرد مدل در سیستم خودآموزی است.
کلاس FeedbackAnalyzer از NeedDetectorBase ارث می‌برد و با بررسی بازخوردهایی مانند نظرات متنی، امتیازدهی کاربران،
و شناسایی کلمات کلیدی منفی یا مشکلات تکراری، نیازهای بهبود را استخراج می‌کند.

این نسخه نهایی و عملیاتی با بهترین مکانیسم‌های هوشمند (مانند تحلیل کلیدواژه و بررسی امتیاز) و کارایی بالا پیاده‌سازی شده است.
"""

import logging
from typing import Dict, Any, List, Optional
import re

from .need_detector_base import NeedDetectorBase


class FeedbackAnalyzer(NeedDetectorBase):
    """
    FeedbackAnalyzer برای تحلیل بازخوردهای دریافتی از کاربران و استخراج نیازهای بهبود عملکرد مدل طراحی شده است.

    ورودی مورد انتظار:
      {
         "feedbacks": List[Dict[str, Any]]
      }
      هر بازخورد می‌تواند شامل:
         - "feedback_text": متن بازخورد (str)
         - "rating": امتیاز داده‌شده توسط کاربر (float یا int؛ مقیاس 1 تا 5)
         - "timestamp": زمان ارائه بازخورد (اختیاری)

    خروجی:
      لیستی از نیازهای بهبود بر اساس تحلیل بازخورد، به‌صورت دیکشنری، مثلاً:
         {
           "need_type": "FEEDBACK",
           "issue": "Low rating",
           "feedback_text": "The response was too slow.",
           "rating": 2,
           "details": "Rating below acceptable threshold and contains keywords like 'slow'."
         }
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        مقداردهی اولیه FeedbackAnalyzer.

        Args:
            config (Optional[Dict[str, Any]]): پیکربندی اختصاصی، شامل:
                - "rating_threshold": امتیاز پایینی که در صورت پایین‌تر بودن آن، نیاز به بهبود گزارش شود (پیش‌فرض: 3).
                - "negative_keywords": لیستی از کلمات کلیدی منفی جهت شناسایی مسائل (پیش‌فرض: ["slow", "error", "inaccurate", "bad"]).
        """
        super().__init__(config=config)
        self.logger = logging.getLogger("FeedbackAnalyzer")
        self.rating_threshold = float(self.config.get("rating_threshold", 3))
        self.negative_keywords = self.config.get("negative_keywords", ["slow", "error", "inaccurate", "bad"])
        # کامپایل یک regex برای کلمات کلیدی جهت بهبود کارایی
        self.keyword_pattern = re.compile("|".join(self.negative_keywords), re.IGNORECASE)
        self.logger.info(f"[FeedbackAnalyzer] Initialized with rating_threshold={self.rating_threshold} "
                         f"and negative_keywords={self.negative_keywords}")

    def detect_needs(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        شناسایی نیازهای بهبود بر اساس بازخوردهای کاربران.

        Args:
            input_data (Dict[str, Any]): دیکشنری شامل:
                - "feedbacks": لیستی از بازخوردها؛ هر بازخورد یک دیکشنری با کلیدهای "feedback_text" و "rating" است.

        Returns:
            List[Dict[str, Any]]: لیستی از نیازهای بهبود شناسایی‌شده.
        """
        if not self.validate_input(input_data):
            self.logger.warning("[FeedbackAnalyzer] Invalid input data.")
            return []

        feedbacks = input_data.get("feedbacks", [])
        if not feedbacks:
            self.logger.info("[FeedbackAnalyzer] No feedbacks provided; no needs detected.")
            return []

        detected_needs = []
        for feedback in feedbacks:
            text = feedback.get("feedback_text", "").strip()
            try:
                rating = float(feedback.get("rating", 5))
            except (TypeError, ValueError):
                rating = 5.0  # در صورت عدم ارائه یا خطا، امتیاز بالا را فرض می‌کنیم

            details = []
            need_detected = False

            # بررسی امتیاز
            if rating < self.rating_threshold:
                need_detected = True
                details.append(f"Rating ({rating}) is below threshold ({self.rating_threshold}).")

            # بررسی کلمات کلیدی منفی در متن بازخورد
            if text and self.keyword_pattern.search(text):
                need_detected = True
                details.append("Feedback contains negative keywords.")

            if need_detected:
                need_info = {
                    "need_type": "FEEDBACK",
                    "issue": "User Feedback Issue",
                    "feedback_text": text,
                    "rating": rating,
                    "details": " ".join(details)
                }
                detected_needs.append(need_info)
                self.logger.debug(f"[FeedbackAnalyzer] Detected feedback need: {need_info}")

        final_needs = self.post_process_needs(detected_needs)
        self.logger.info(f"[FeedbackAnalyzer] Total feedback needs detected: {len(final_needs)}")
        return final_needs
