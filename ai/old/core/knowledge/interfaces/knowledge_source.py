from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class KnowledgeSourceType(Enum):
    """انواع منابع دانشی که سیستم می‌تواند با آنها کار کند"""
    CHAT_GPT = "chat_gpt"  # منبع ChatGPT
    WIKIPEDIA = "wikipedia"  # منبع ویکی‌پدیا
    LOCAL_FILE = "local_file"  # فایل‌های محلی
    REMOTE_FILE = "remote_file"  # فایل‌های از راه دور
    DATABASE = "database"  # پایگاه داده
    API = "api"  # سرویس‌های API
    CUSTOM = "custom"  # منابع سفارشی


class KnowledgePriority(Enum):
    """اولویت‌های یادگیری برای محتوای دانشی"""
    CRITICAL = 1  # دانش حیاتی که باید فوراً یاد گرفته شود
    HIGH = 2  # دانش با اولویت بالا
    MEDIUM = 3  # دانش با اولویت متوسط
    LOW = 4  # دانش با اولویت پایین
    BACKGROUND = 5  # دانش زمینه‌ای که می‌تواند در زمان خلوت یاد گرفته شود


@dataclass
class LearningContext:
    """زمینه و شرایط یادگیری"""
    focus_areas: List[str]  # حوزه‌های تمرکز
    learning_mode: str  # حالت یادگیری (عمیق، سریع، مروری)
    required_confidence: float  # حداقل اطمینان مورد نیاز
    max_processing_time: Optional[int]  # حداکثر زمان پردازش (به ثانیه)
    dependencies: List[str]  # وابستگی‌های دانشی
    constraints: Dict[str, Any]  # محدودیت‌های خاص


@dataclass
class KnowledgeMetadata:
    """متادیتای مربوط به هر قطعه دانش"""
    source_id: str  # شناسه یکتای منبع
    source_type: KnowledgeSourceType  # نوع منبع دانش
    created_at: datetime  # زمان ایجاد
    updated_at: datetime  # زمان آخرین به‌روزرسانی
    priority: KnowledgePriority  # اولویت یادگیری
    confidence_score: float  # نمره اطمینان (0 تا 1)
    validation_status: bool  # وضعیت اعتبارسنجی
    learning_progress: float  # پیشرفت یادگیری (0 تا 1)
    tags: List[str]  # برچسب‌های موضوعی
    language: str  # زبان محتوا
    size_bytes: int  # حجم محتوا
    checksum: str  # هش محتوا برای تشخیص تغییرات

    def to_dict(self) -> Dict:
        """تبدیل متادیتا به دیکشنری"""
        return {
            'source_id': self.source_id,
            'source_type': self.source_type.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'priority': self.priority.value,
            'confidence_score': self.confidence_score,
            'validation_status': self.validation_status,
            'learning_progress': self.learning_progress,
            'tags': self.tags,
            'language': self.language,
            'size_bytes': self.size_bytes,
            'checksum': self.checksum
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'KnowledgeMetadata':
        """ایجاد متادیتا از دیکشنری"""
        return cls(
            source_id=data['source_id'],
            source_type=KnowledgeSourceType(data['source_type']),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            priority=KnowledgePriority(data['priority']),
            confidence_score=data['confidence_score'],
            validation_status=data['validation_status'],
            learning_progress=data['learning_progress'],
            tags=data['tags'],
            language=data['language'],
            size_bytes=data['size_bytes'],
            checksum=data['checksum']
        )


class KnowledgeSource(ABC):
    """
    کلاس پایه برای تمام منابع دانش

    این کلاس تعریف می‌کند که هر منبع دانش چه قابلیت‌هایی باید داشته باشد.
    هر منبع جدید باید این کلاس را پیاده‌سازی کند.
    """

    def __init__(self, source_config: Dict[str, Any]):
        """
        مقداردهی اولیه منبع دانش

        Args:
            source_config: تنظیمات مورد نیاز برای راه‌اندازی منبع
        """
        self.config = source_config
        self.metadata = None
        self._initialized = False
        self._local_cache = {}  # کش موقت برای بهبود کارایی در حالت محلی
        logger.info(f"Initializing knowledge source with config: {source_config}")

    @abstractmethod
    async def initialize(self) -> bool:
        """
        راه‌اندازی و آماده‌سازی منبع دانش

        Returns:
            موفقیت‌آمیز بودن راه‌اندازی
        """
        pass

    @abstractmethod
    async def get_knowledge(self, query: str, context: Optional[LearningContext] = None) -> Dict[str, Any]:
        """
        دریافت دانش از منبع

        Args:
            query: پرس‌وجوی دانش مورد نیاز
            context: زمینه یادگیری (اختیاری)

        Returns:
            دیکشنری حاوی دانش و متادیتای مربوطه
        """
        pass

    @abstractmethod
    async def validate_knowledge(self, knowledge_data: Dict[str, Any]) -> bool:
        """
        اعتبارسنجی دانش دریافت شده

        Args:
            knowledge_data: داده‌های دانش برای اعتبارسنجی

        Returns:
            نتیجه اعتبارسنجی
        """
        pass

    @abstractmethod
    async def update_learning_progress(self, knowledge_id: str, progress: float) -> None:
        """
        به‌روزرسانی پیشرفت یادگیری

        Args:
            knowledge_id: شناسه دانش
            progress: میزان پیشرفت (0 تا 1)
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """پاکسازی منابع و اتمام کار با منبع دانش"""
        pass

    async def __aenter__(self):
        """امکان استفاده با context manager"""
        if not self._initialized:
            await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """پاکسازی خودکار منابع"""
        await self.cleanup()

    def _validate_config(self) -> bool:
        """اعتبارسنجی تنظیمات منبع دانش"""
        required_fields = ['source_id', 'source_type', 'max_retries']
        return all(field in self.config for field in required_fields)

    def _handle_error(self, error: Exception, context: str) -> None:
        """مدیریت و ثبت خطاها"""
        logger.error(f"Error in {context}: {str(error)}", exc_info=True)
        # در حالت محلی، خطاها را مستقیماً نمایش می‌دهیم
        if self.config.get('environment') == 'local':
            print(f"Error: {str(error)}")