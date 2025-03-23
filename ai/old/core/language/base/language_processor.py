from abc import ABC, abstractmethod
from types import TracebackType
from typing import TypeVar, Generic, Dict, Any, List, Optional, Type
from datetime import datetime
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)
T = TypeVar('T')


@dataclass
class ProcessingResult(Generic[T]):
    text: str
    tokens: List[str]
    features: T  # Generic features
    confidence: float
    language: str
    analysis_time: datetime
    metadata: Dict[str, Any]

    def is_confident(self, threshold: float = 0.7) -> bool:
        """بررسی کافی بودن اطمینان به نتایج"""
        return self.confidence >= threshold

    def merge_with(self, other: 'ProcessingResult[T]') -> 'ProcessingResult[T]':
        """ترکیب با نتیجه دیگر با اولویت نتیجه با اطمینان بیشتر"""
        if other.confidence > self.confidence:
            base_result = other
            additional = self
        else:
            base_result = self
            additional = other

        merged_features = {**additional.features, **base_result.features}
        merged_metadata = {**additional.metadata, **base_result.metadata}

        return ProcessingResult(
            text=base_result.text,
            tokens=base_result.tokens,
            confidence=max(self.confidence, other.confidence),
            language=base_result.language,
            analysis_time=datetime.now(),
            features=merged_features,
            metadata=merged_metadata
        )


class LanguageProcessor(ABC):
    """
    کلاس پایه برای پردازش زبان

    این کلاس یک چارچوب اصلی برای پردازش متن در هر زبانی فراهم می‌کند.
    کلاس‌های فرزند باید متدهای اصلی را پیاده‌سازی کنند تا عملکرد
    مورد نظر برای آن زبان خاص را فراهم کنند.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        مقداردهی اولیه پردازشگر زبان

        Args:
            config: تنظیمات اولیه پردازشگر (اختیاری)
        """
        self.config = config or {}
        self._initialized = False
        self._last_process_time = None
        self.knowledge_base = {}  # پایگاه دانش اولیه

    async def __aenter__(self) -> 'LanguageProcessor':
        await self.initialize()
        return self

    async def __aexit__(self,
                        exc_type: Optional[Type[BaseException]],
                        exc_val: Optional[BaseException],
                        exc_tb: Optional[TracebackType]
                        ) -> None:
        await self.cleanup()

    async def cleanup(self) -> None:
        """پاکسازی منابع"""
        self._initialized = False
        self.knowledge_base.clear()

    @abstractmethod
    async def initialize(self) -> bool:
        """
        راه‌اندازی اولیه پردازشگر

        Returns:
            موفقیت در راه‌اندازی
        """
        pass

    @abstractmethod
    async def process(self, text: str) -> ProcessingResult:
        """
        پردازش متن و استخراج ویژگی‌ها

        Args:
            text: متن ورودی

        Returns:
            نتایج پردازش متن
        """
        pass

    @abstractmethod
    async def learn(self, text: str, analysis: ProcessingResult) -> None:
        """
        یادگیری از تحلیل متن

        Args:
            text: متن اصلی
            analysis: نتایج تحلیل متن
        """
        pass

    @abstractmethod
    async def validate_text(self, text: str) -> bool:
        """
        اعتبارسنجی متن از نظر زبان مورد نظر

        Args:
            text: متن برای بررسی

        Returns:
            معتبر بودن متن
        """
        pass

    async def analyze_deeply(self, text: str) -> ProcessingResult:
        """
        تحلیل عمیق متن با استفاده از تمام منابع موجود

        این متد از ترکیب دانش داخلی و منابع خارجی برای
        بهترین تحلیل ممکن استفاده می‌کند.

        Args:
            text: متن برای تحلیل

        Returns:
            نتایج کامل تحلیل
        """
        try:
            # بررسی اعتبار متن
            if not await self.validate_text(text):
                raise ValueError("متن برای این زبان معتبر نیست")

            # پردازش اولیه با دانش داخلی
            initial_result = await self.process(text)

            # اگر اطمینان کافی نداریم، از منابع خارجی کمک می‌گیریم
            if not initial_result.is_confident():
                external_result = await self._get_external_analysis(text)
                if external_result:
                    # ترکیب نتایج داخلی و خارجی
                    final_result = initial_result.merge_with(external_result)
                    # یادگیری از تحلیل جدید
                    await self.learn(text, external_result)
                    return final_result

            return initial_result

        except Exception as e:
            logger.error(f"Error in deep analysis: {str(e)}")
            raise

    @abstractmethod
    async def _get_external_analysis(self, text: str) -> Optional[ProcessingResult]:
        """
        دریافت تحلیل از منابع خارجی

        Args:
            text: متن برای تحلیل

        Returns:
            نتایج تحلیل خارجی یا None در صورت عدم موفقیت
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """
        دریافت آمار عملکرد پردازشگر

        Returns:
            دیکشنری حاوی آمار عملکرد
        """
        return {
            'initialized': self._initialized,
            'last_process_time': self._last_process_time,
            'knowledge_base_size': len(self.knowledge_base),
            'config': self.config
        }