# persian/language_processors/contextual/intent_analyzer.py
"""
ماژول intent_analyzer.py

این ماژول شامل کلاس IntentAnalyzer است که وظیفه تحلیل نیت پیام‌های ورودی به‌صورت جامع و چند بعدی را بر عهده دارد.
این کلاس با استفاده از ابزارهای نرمال‌سازی (TextNormalizer) و توکنیزیشن (Tokenizer) از پوشه‌ی utils،
الگوهای regex تعریف‌شده و الگوریتم‌های تطبیقی، نیت پیام (intent type)، موضوعات کلیدی، فوریت، حالت احساسی
و وابستگی به زمینه (context dependency) را استخراج می‌کند.
"""

import re
import time
from collections import Counter
from typing import List, Dict, Any, Optional

from ..utils.text_normalization import TextNormalizer
from ..utils.tokenization import Tokenizer
from .config import get_confidence_levels

# در این نسخه می‌توان در آینده مدل‌های هوشمند (مانند smart_model یا teacher) را به‌عنوان ورودی به کلاس اضافه کرد
# تا تحلیل به صورت مبتنی بر یادگیری انجام شود. در حال حاضر، از الگوریتم‌های مبتنی بر الگو استفاده می‌شود.


class IntentAnalyzer:
    """
    کلاس IntentAnalyzer

    وظیفه این کلاس تحلیل نیت پیام (intent) به همراه استخراج موضوعات کلیدی، فوریت، حالت احساسی و وابستگی به زمینه را دارد.
    خروجی نهایی به صورت یک دیکشنری یکپارچه ارائه می‌شود.
    """
    def __init__(self, model: Optional[Any] = None):
        """
        مقداردهی اولیه کلاس IntentAnalyzer.

        Args:
            model (Optional[Any]): شی مدل هوشمند (اختیاری) جهت بهبود تحلیل نیت؛ در صورت عدم وجود از تحلیل مبتنی بر الگو استفاده می‌شود.
        """
        self.normalizer = TextNormalizer()
        self.tokenizer = Tokenizer()
        self.model = model  # امکان گسترش در آینده برای استفاده از مدل‌های یادگیری

        # تعریف الگوهای اصلی جهت تعیین نوع نیت
        self.question_patterns = [
            r'\?', r'؟',
            r'\b(چه|چرا|کجا|چگونه|چطور|کی|کدام)\b',
            r'\b(آیا|مگر)\b.*\?'
        ]
        self.request_patterns = [
            r'\b(لطفاً?|خواهش|تمنا)\b',
            r'\b(می‌توانی|می‌شود|میشه|ممکنه)\b',
            r'\b(باید|نیاز دارم|میخواهم|می‌خواهم)\b'
        ]
        self.command_patterns = [
            r'^[^،؛.!?؟]+[!.]',
            r'\b(انجام بده|اجرا کن|بساز|پیدا کن|محاسبه کن)\b'
        ]
        self.inform_patterns = [
            r'\b(می‌خواستم بگم|اطلاع بدم|بدانید که|گزارش می‌دهم)\b',
            r'\b(فکر می‌کنم|به نظرم|معتقدم|حدس می‌زنم)\b'
        ]
        self.urgent_keywords = ["فوری", "اضطراری", "سریع", "عجله", "بلافاصله", "آنی", "هر چه زودتر", "فوراً", "همین الان", "بدون تأخیر", "وقت کم", "ضرب‌الاجل", "مهلت"]

        # کلمات احساسی مثبت و منفی
        self.positive_keywords = {"خوب", "عالی", "فوق‌العاده", "خوشحال", "راضی", "خرسند", "سپاسگزار", "ممنون", "متشکر", "لذت", "عشق", "دوست داشتن", "امیدوار", "موفق", "پیروز", "مثبت"}
        self.negative_keywords = {"بد", "افتضاح", "ناراحت", "عصبانی", "خشمگین", "متأسف", "ناراضی", "غمگین", "ناامید", "شکست", "ناکامی", "نفرت", "متنفر", "خسته", "ناموفق", "منفی"}

        # نشانه‌های احساسی (با امتیازدهی)
        self.emotional_markers = {
            "!": 0.2, "!!": 0.4, "!!!": 0.6,
            "؟؟": 0.3,
            "😊": 0.5, "😢": 0.5, "😡": 0.6
        }

    def analyze(self, message: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        تحلیل نیت پیام ورودی به همراه استخراج موضوعات، فوریت، حالت احساسی و وابستگی به زمینه.
        در صورت ارائه‌ی پیام‌های زمینه‌ای (context)، می‌توان به بررسی وابستگی نیز پرداخت.

        Args:
            message (str): متن پیام ورودی
            context (Optional[List[str]]): لیستی از پیام‌های قبلی یا زمینه (اختیاری)

        Returns:
            Dict[str, Any]: دیکشنری شامل:
                - intent_type: نوع نیت (QUESTION, REQUEST, COMMAND, INFORM, STATEMENT)
                - topics: لیست موضوعات کلیدی استخراج‌شده
                - urgency: امتیاز فوریت (0 تا 1)
                - emotional_state: دیکشنری شامل امتیازهای positive، negative و neutral
                - context_dependency: اطلاعات وابستگی به زمینه (is_context_dependent و dependency_level)
        """
        normalized_message = self.normalizer.normalize(message)
        tokens = self.tokenizer.tokenize_words(normalized_message)

        # در صورت وجود مدل هوشمند، می‌توان از آن استفاده کرد (در اینجا به صورت اختیاری)
        if self.model:
            # مثال: خروجی مدل می‌تواند شامل intent، topics و ... باشد.
            smart_output = self.model.analyze_intent(normalized_message, context)
            if smart_output:
                return smart_output

        intent_type = self._determine_intent_type(normalized_message)
        topics = self._extract_topics(tokens)
        urgency = self._determine_urgency(normalized_message)
        emotional_state = self._determine_emotional_state(normalized_message)
        context_dependency = self._determine_context_dependency(normalized_message, context)

        return {
            "intent_type": intent_type,
            "topics": topics,
            "urgency": urgency,
            "emotional_state": emotional_state,
            "context_dependency": context_dependency
        }

    def _determine_intent_type(self, message: str) -> str:
        """
        تعیین نوع نیت پیام بر اساس الگوهای از پیش تعریف‌شده.

        Args:
            message (str): پیام نرمال‌شده

        Returns:
            str: یکی از انواع "QUESTION"، "REQUEST"، "COMMAND"، "INFORM" یا "STATEMENT"
        """
        for pattern in self.question_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return "QUESTION"
        for pattern in self.request_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return "REQUEST"
        for pattern in self.command_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return "COMMAND"
        for pattern in self.inform_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return "INFORM"
        # اگر هیچ کدام از الگوها تطبیق ندادند، بررسی احساس نیز انجام می‌شود
        if self._has_emotion(message):
            return "EMOTIONAL"
        return "STATEMENT"

    def _extract_topics(self, tokens: List[str]) -> List[str]:
        """
        استخراج موضوعات کلیدی از پیام بر اساس فراوانی کلمات.

        Args:
            tokens (List[str]): لیست کلمات استخراج‌شده از پیام

        Returns:
            List[str]: لیستی از کلمات پرتکرار به‌عنوان موضوعات
        """
        # حذف کلمات ایست ساده فارسی
        stop_words = {"و", "در", "به", "از", "که", "این", "است", "را", "با", "های", "برای", "آن"}
        filtered_tokens = [token for token in tokens if token.lower() not in stop_words and len(token) > 2]
        if not filtered_tokens:
            return []
        frequency = Counter(filtered_tokens)
        common = frequency.most_common(3)
        return [word for word, count in common]

    def _determine_urgency(self, message: str) -> float:
        """
        تعیین امتیاز فوریت پیام بر اساس حضور کلمات و عبارات فوریت.

        Args:
            message (str): پیام نرمال‌شده

        Returns:
            float: امتیاز فوریت (بین 0 تا 1)
        """
        score = 0.0
        lower_message = message.lower()
        for keyword in self.urgent_keywords:
            if keyword.lower() in lower_message:
                score += 0.2
        # محدود کردن به بیشترین مقدار 1.0
        return min(round(score, 2), 1.0)

    def _determine_emotional_state(self, message: str) -> Dict[str, float]:
        """
        تحلیل حالت احساسی پیام بر اساس کلمات و نشانه‌های احساسی.

        Args:
            message (str): پیام نرمال‌شده

        Returns:
            Dict[str, float]: شامل امتیازهای positive، negative و neutral
        """
        pos_score = 0.0
        neg_score = 0.0
        tokens = self.tokenizer.tokenize_words(message.lower())
        # امتیازدهی براساس کلمات مثبت و منفی
        for token in tokens:
            if token in self.positive_keywords:
                pos_score += 1.0
            if token in self.negative_keywords:
                neg_score += 1.0
        total = pos_score + neg_score
        if total > 0:
            pos_ratio = pos_score / total
            neg_ratio = neg_score / total
        else:
            pos_ratio = neg_ratio = 0.0
        neutral_ratio = 1.0 - (pos_ratio + neg_ratio)
        return {
            "positive": round(pos_ratio, 2),
            "negative": round(neg_ratio, 2),
            "neutral": round(neutral_ratio, 2)
        }

    def _determine_context_dependency(self, message: str, context: Optional[List[str]]) -> Dict[str, Any]:
        """
        ارزیابی وابستگی پیام به زمینه (context)؛ اگر پیام کوتاه یا دارای ضمایر بدون مرجع است،
        احتمال وابستگی به مکالمه قبلی افزایش می‌یابد.

        Args:
            message (str): پیام نرمال‌شده
            context (Optional[List[str]]): لیستی از پیام‌های قبلی یا زمینه

        Returns:
            Dict[str, Any]: شامل:
                - is_context_dependent (bool)
                - dependency_level (float): مقدار بین 0 تا 1
                - referenced_context (List[str]): لیستی از پیام‌های زمینه‌ای مرتبط (در صورت وجود)
        """
        result = {
            "is_context_dependent": False,
            "dependency_level": 0.0,
            "referenced_context": []
        }
        if not context or len(context) == 0:
            return result

        # اگر پیام بسیار کوتاه است، احتمال وابستگی بالا است
        if len(message.split()) < 5:
            result["is_context_dependent"] = True
            result["dependency_level"] = 0.6

        # بررسی شباهت با پیام‌های زمینه
        similarities = []
        for ctx_msg in context:
            sim = self._calculate_similarity(message, ctx_msg)
            similarities.append(sim)
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        # اگر شباهت میانگین بالا بود، وابستگی بیشتر است
        if avg_similarity > 0.3:
            result["is_context_dependent"] = True
            result["dependency_level"] = round(min(avg_similarity + 0.3, 1.0), 2)
            # انتخاب پیام‌های زمینه‌ای با شباهت بالا
            referenced = [ctx_msg for ctx_msg, sim in zip(context, similarities) if sim > 0.3]
            result["referenced_context"] = referenced

        return result

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        محاسبه شباهت دو متن با استفاده از الگوریتم SequenceMatcher.

        Args:
            text1 (str): متن اول
            text2 (str): متن دوم

        Returns:
            float: مقدار شباهت (0 تا 1)
        """
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1, text2).ratio()

    def _has_emotion(self, message: str) -> bool:
        """
        بررسی وجود نشانه‌های احساسی در پیام.

        Args:
            message (str): پیام نرمال‌شده

        Returns:
            bool: True در صورت وجود احساس، در غیر این صورت False
        """
        # بررسی وجود ایموجی
        emoji_pattern = re.compile(
            "[" 
            "\U0001F600-\U0001F64F"  # صورتک‌ها
            "\U0001F300-\U0001F5FF"  # نمادها
            "\U0001F680-\U0001F6FF"  # حمل و نقل
            "\U0001F700-\U0001F77F"  # علائم
            "\U0001F780-\U0001F7FF"  
            "\U0001F800-\U0001F8FF"  
            "\U0001F900-\U0001F9FF"  
            "\U0001FA00-\U0001FA6F"  
            "\U0001FA70-\U0001FAFF"  
            "\U00002702-\U000027B0"  
            "\U000024C2-\U0001F251"
            "]+"
        )
        if emoji_pattern.search(message):
            return True
        # بررسی تکرار کاراکتر
        if re.search(r'(.)\1{2,}', message):
            return True
        return False

# نمونه تست مستقل
if __name__ == "__main__":
    sample_text = "سلام! لطفاً راهنمایی کن چطور می‌توانم این مشکل رو سریع حل کنم؟"
    analyzer = IntentAnalyzer()
    result = analyzer.analyze(sample_text, context=["من امروز خیلی خسته هستم", "مشکل از قسمت شبکه است"])
    print("نتیجه تحلیل نیت:", result)
