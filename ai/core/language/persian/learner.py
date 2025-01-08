"""
Persian Language Learner
----------------------
این ماژول مسئول یادگیری و بهبود مستمر در درک و پردازش زبان فارسی است.
سیستم از طریق تحلیل نمونه‌های جدید، استخراج الگوها و به‌روزرسانی پایگاه دانش،
به تدریج توانایی خود در درک زبان فارسی را افزایش می‌دهد.
"""

from typing import Dict, List, Any, Optional, Set
import logging
from datetime import datetime
import asyncio
import numpy as np
from pathlib import Path
import json

from ..base.language_learner import LanguageLearner, LearningExample
from ..external.parsbert_adapter import ParsBERTAdapter
from ..external.hazm_adapter import HazmAdapter

logger = logging.getLogger(__name__)


class PatternExtractor:
    """کلاس کمکی برای استخراج الگوهای زبانی از متن فارسی"""

    def __init__(self):
        self.common_patterns: Dict[str, int] = {}  # الگو -> تعداد تکرار
        self.min_pattern_length = 2  # حداقل طول الگو
        self.min_occurrence = 3  # حداقل تعداد تکرار برای معتبر بودن الگو

    def add_pattern(self, pattern: str) -> None:
        """اضافه کردن یک الگوی جدید یا به‌روزرسانی تعداد تکرار"""
        if len(pattern.split()) >= self.min_pattern_length:
            self.common_patterns[pattern] = self.common_patterns.get(pattern, 0) + 1

    def get_valid_patterns(self) -> Set[str]:
        """دریافت الگوهای معتبر (با تکرار کافی)"""
        return {pattern for pattern, count in self.common_patterns.items()
                if count >= self.min_occurrence}


class PersianLearner(LanguageLearner):
    """
    یادگیرنده زبان فارسی

    این کلاس با تحلیل نمونه‌های متنی، الگوهای زبانی را شناسایی می‌کند و
    دانش خود را برای بهبود پردازش زبان فارسی گسترش می‌دهد.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """راه‌اندازی یادگیرنده با تنظیمات اختیاری"""
        super().__init__(config)
        self.parsbert = ParsBERTAdapter()
        self.hazm = HazmAdapter()
        self.pattern_extractor = PatternExtractor()
        self._semantic_cache: Dict[str, np.ndarray] = {}  # کش بردارهای معنایی
        self.learning_stats = {
            'total_examples': 0,
            'verified_patterns': 0,
            'semantic_vectors': 0
        }

    async def learn(self, text: str, features: Dict[str, Any], source: str) -> None:
        """
        یادگیری از یک نمونه جدید متن فارسی

        در این متد، سیستم سعی می‌کند الگوهای جدید را شناسایی کند،
        بردارهای معنایی را ذخیره کند و دانش خود را گسترش دهد.

        Args:
            text: متن نمونه
            features: ویژگی‌های استخراج شده از متن
            source: منبع نمونه (مثل 'external', 'human', 'self')
        """
        try:
            # تحلیل و استخراج اطلاعات از نمونه جدید
            hazm_result = await self.hazm.process_text(text)
            parsbert_result = await self.parsbert.process_text(text)

            if not hazm_result or not parsbert_result:
                logger.warning(f"Could not process text for learning: {text[:50]}...")
                return

            # استخراج و ذخیره الگوها
            patterns = await self.extract_patterns(text)
            for pattern in patterns:
                self.pattern_extractor.add_pattern(pattern)

            # ذخیره بردار معنایی
            text_vector = parsbert_result.sentence_embedding.numpy()
            self._semantic_cache[text] = text_vector

            # ایجاد نمونه یادگیری
            example = LearningExample(
                text=text,
                features={
                    **features,
                    'patterns': patterns,
                    'semantic_vector': text_vector,
                    'pos_tags': hazm_result.pos_tags
                },
                source=source,
                confidence=self._calculate_learning_confidence(
                    patterns, text_vector, hazm_result.pos_tags
                ),
                timestamp=datetime.now()
            )

            # افزودن به پایگاه دانش
            await self.add_example(example)

            # به‌روزرسانی آمار
            self.learning_stats['total_examples'] += 1
            self.learning_stats['semantic_vectors'] = len(self._semantic_cache)
            self.learning_stats['verified_patterns'] = len(
                self.pattern_extractor.get_valid_patterns()
            )

        except Exception as e:
            logger.error(f"Error in learning process: {e}")

    async def extract_patterns(self, text: str) -> List[str]:
        """
        استخراج الگوهای زبانی از متن

        این متد الگوهای تکرارشونده و معنادار را از متن استخراج می‌کند.
        از تحلیل نحوی Hazm برای شناسایی ساختارهای معنادار استفاده می‌شود.

        Args:
            text: متن برای تحلیل

        Returns:
            لیست الگوهای یافت شده
        """
        patterns = []
        hazm_result = await self.hazm.process_text(text)

        if not hazm_result:
            return patterns

        # استخراج الگوهای نحوی
        current_pattern = []
        for word, tag in hazm_result.pos_tags:
            if tag.startswith(('N', 'V', 'ADJ')):  # اسم، فعل، صفت
                current_pattern.append(f"{word}({tag})")
            else:
                if current_pattern:
                    patterns.append(' '.join(current_pattern))
                    current_pattern = []

        if current_pattern:
            patterns.append(' '.join(current_pattern))

        return patterns

    async def _calculate_similarity(self, patterns: List[str],
                                    example: LearningExample) -> float:
        """
        محاسبه میزان شباهت بین الگوها و یک نمونه

        از ترکیب شباهت معنایی و ساختاری برای محاسبه شباهت کلی استفاده می‌شود.

        Args:
            patterns: الگوهای مورد نظر
            example: نمونه برای مقایسه

        Returns:
            میزان شباهت (0 تا 1)
        """
        # محاسبه شباهت الگوها
        pattern_similarity = len(
            set(patterns) & set(example.features.get('patterns', []))
        ) / max(len(patterns), len(example.features.get('patterns', [])))

        # محاسبه شباهت معنایی
        if 'semantic_vector' in example.features:
            current_vector = example.features['semantic_vector']
            semantic_similarity = float(np.dot(
                patterns[0], current_vector
            ) / (np.linalg.norm(patterns[0]) * np.linalg.norm(current_vector)))
        else:
            semantic_similarity = 0.0

        # ترکیب شباهت‌ها با وزن مناسب
        return 0.7 * semantic_similarity + 0.3 * pattern_similarity

    def _calculate_learning_confidence(self, patterns: List[str],
                                       vector: np.ndarray,
                                       pos_tags: List[tuple]) -> float:
        """
        محاسبه میزان اطمینان به یادگیری

        بر اساس کیفیت الگوها، بردار معنایی و تحلیل نحوی، میزان اطمینان
        به صحت یادگیری را محاسبه می‌کند.

        Args:
            patterns: الگوهای استخراج شده
            vector: بردار معنایی
            pos_tags: برچسب‌های دستوری

        Returns:
            میزان اطمینان (0 تا 1)
        """
        # محاسبه کیفیت الگوها
        pattern_quality = min(1.0, len(patterns) / 10)

        # محاسبه کیفیت بردار معنایی
        vector_quality = float(np.mean(np.abs(vector)))

        # محاسبه کیفیت تحلیل نحوی
        pos_quality = sum(1 for _, tag in pos_tags if tag != 'UNKNOWN') / len(pos_tags)

        # ترکیب معیارها با وزن‌های مختلف
        return (0.4 * pattern_quality +
                0.4 * vector_quality +
                0.2 * pos_quality)

    def get_learning_stats(self) -> Dict[str, Any]:
        """دریافت آمار یادگیری"""
        return {
            **self.learning_stats,
            'cache_size': len(self._semantic_cache),
            'knowledge_base_size': len(self.knowledge_base)
        }