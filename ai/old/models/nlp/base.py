"""
رابط‌ها و کلاس‌های پایه برای ماژول NLP

این ماژول پایه‌ای‌ترین لایه از معماری NLP را تعریف می‌کند. طبق اصول معماری تمیز،
این لایه شامل رابط‌ها و موجودیت‌های اصلی کسب و کار است و به هیچ لایه بیرونی
وابستگی ندارد.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
from . import NLPModelConfig, SupportedLanguage


@dataclass
class TextInput:
    """موجودیت ورودی متنی"""
    text: str
    language: SupportedLanguage
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = datetime.now()

    def validate(self) -> bool:
        """اعتبارسنجی ورودی"""
        return bool(self.text and self.text.strip())


@dataclass
class ProcessedText:
    """موجودیت متن پردازش شده"""
    original_input: TextInput
    processed_text: str
    tokens: List[str]
    embeddings: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0


class TextPreprocessor(ABC):
    """رابط پیش‌پردازش متن"""

    @abstractmethod
    async def preprocess(self, text_input: TextInput) -> ProcessedText:
        """پیش‌پردازش متن ورودی"""
        pass

    @abstractmethod
    async def normalize(self, text: str) -> str:
        """نرمال‌سازی متن"""
        pass

    @abstractmethod
    async def tokenize(self, text: str) -> List[str]:
        """توکنایز کردن متن"""
        pass


class TextAnalyzer(ABC):
    """رابط تحلیل متن"""

    @abstractmethod
    async def analyze_sentiment(self, processed_text: ProcessedText) -> Dict[str, float]:
        """تحلیل احساسات متن"""
        pass

    @abstractmethod
    async def extract_entities(self, processed_text: ProcessedText) -> List[Dict[str, Any]]:
        """استخراج موجودیت‌ها"""
        pass

    @abstractmethod
    async def classify_text(self, processed_text: ProcessedText) -> Dict[str, float]:
        """دسته‌بندی متن"""
        pass


class ModelRepository(ABC):
    """رابط مخزن مدل - برای مدیریت ذخیره و بازیابی مدل‌ها"""

    @abstractmethod
    async def save_model(self, model_path: str, metadata: Dict[str, Any]) -> bool:
        """ذخیره مدل"""
        pass

    @abstractmethod
    async def load_model(self, model_id: str) -> Any:
        """بارگذاری مدل"""
        pass

    @abstractmethod
    async def list_models(self) -> List[Dict[str, Any]]:
        """لیست تمام مدل‌های موجود"""
        pass


class NLPBaseModel(ABC):
    """کلاس پایه برای مدل‌های NLP"""

    def __init__(self, config: NLPModelConfig):
        self.config = config
        self._preprocessor: Optional[TextPreprocessor] = None
        self._analyzer: Optional[TextAnalyzer] = None
        self._repository: Optional[ModelRepository] = None

    @abstractmethod
    async def initialize(self) -> None:
        """راه‌اندازی اولیه مدل"""
        pass

    @abstractmethod
    async def process(self, text_input: TextInput) -> ProcessedText:
        """پردازش متن ورودی"""
        if not text_input.validate():
            raise ValueError("Invalid input text")
        pass

    @abstractmethod
    async def train(self, training_data: List[TextInput]) -> None:
        """آموزش مدل"""
        pass

    @abstractmethod
    async def evaluate(self, test_data: List[TextInput]) -> Dict[str, float]:
        """ارزیابی مدل"""
        pass

    @abstractmethod
    async def save(self) -> bool:
        """ذخیره وضعیت مدل"""
        pass

    @abstractmethod
    async def load(self) -> bool:
        """بارگذاری وضعیت مدل"""
        pass


class NLPModelFactory:
    """کارخانه ساخت مدل‌های NLP"""

    @staticmethod
    async def create_model(config: NLPModelConfig) -> NLPBaseModel:
        """ایجاد نمونه جدید از مدل NLP با توجه به تنظیمات"""
        # این متد در پیاده‌سازی‌های بعدی تکمیل می‌شود
        pass