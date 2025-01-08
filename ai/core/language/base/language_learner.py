from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import logging
from pathlib import Path
import json
import asyncio
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LearningExample:
    """نمونه‌های یادگیری برای ذخیره در پایگاه دانش"""

    text: str  # متن اصلی
    features: Dict[str, Any]  # ویژگی‌های استخراج شده
    source: str  # منبع یادگیری (مثلاً "external", "self", "human")
    confidence: float  # میزان اطمینان به صحت (0 تا 1)
    timestamp: datetime  # زمان یادگیری
    usage_count: int = 0  # تعداد دفعات استفاده
    verified: bool = False  # تأیید شده توسط منبع معتبر

    def update_usage(self) -> None:
        """به‌روزرسانی آمار استفاده"""
        self.usage_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """تبدیل به دیکشنری برای ذخیره‌سازی"""
        return {
            'text': self.text,
            'features': self.features,
            'source': self.source,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'usage_count': self.usage_count,
            'verified': self.verified
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningExample':
        """ساخت نمونه از دیکشنری"""
        return cls(
            text=data['text'],
            features=data['features'],
            source=data['source'],
            confidence=data['confidence'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            usage_count=data['usage_count'],
            verified=data['verified']
        )


class LanguageLearner(ABC):
    """
    کلاس پایه برای یادگیری زبان

    این کلاس وظیفه یادگیری و مدیریت دانش زبانی را بر عهده دارد.
    از تجربیات جدید یاد می‌گیرد و می‌تواند این دانش را برای تحلیل‌های
    آینده به کار گیرد.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        مقداردهی اولیه یادگیرنده

        Args:
            config: تنظیمات اولیه (اختیاری)
        """
        self.config = config or {}
        self.knowledge_base: Dict[str, LearningExample] = {}
        self.patterns: Set[str] = set()
        self.save_path = Path(self.config.get('save_path', 'knowledge_base'))
        self.save_path.mkdir(parents=True, exist_ok=True)
        self._learning_lock = asyncio.Lock()
        self.last_save = datetime.now()

    async def initialize(self) -> bool:
        """
        راه‌اندازی یادگیرنده و بارگذاری دانش قبلی

        Returns:
            موفقیت در راه‌اندازی
        """
        try:
            await self.load_knowledge()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize learner: {e}")
            return False

    @abstractmethod
    async def learn(self, text: str, features: Dict[str, Any], source: str) -> None:
        """
        یادگیری از یک نمونه جدید

        Args:
            text: متن نمونه
            features: ویژگی‌های استخراج شده
            source: منبع یادگیری
        """
        pass

    @abstractmethod
    async def extract_patterns(self, text: str) -> List[str]:
        """
        استخراج الگوهای زبانی از متن

        Args:
            text: متن برای تحلیل

        Returns:
            لیست الگوهای یافت شده
        """
        pass

    async def add_example(self, example: LearningExample) -> None:
        """
        افزودن یک نمونه جدید به پایگاه دانش

        Args:
            example: نمونه یادگیری
        """
        async with self._learning_lock:
            key = self._generate_key(example.text)

            if key in self.knowledge_base:
                # به‌روزرسانی نمونه موجود
                existing = self.knowledge_base[key]
                if example.confidence > existing.confidence:
                    self.knowledge_base[key] = example
            else:
                # افزودن نمونه جدید
                self.knowledge_base[key] = example

            # استخراج و ذخیره الگوها
            patterns = await self.extract_patterns(example.text)
            self.patterns.update(patterns)

            # ذخیره دوره‌ای دانش
            await self._auto_save()

    async def find_similar(self, text: str, threshold: float = 0.7) -> List[LearningExample]:
        """
        یافتن نمونه‌های مشابه در پایگاه دانش

        Args:
            text: متن برای جستجو
            threshold: حداقل میزان شباهت (0 تا 1)

        Returns:
            لیست نمونه‌های مشابه
        """
        similar_examples = []
        patterns = await self.extract_patterns(text)

        for example in self.knowledge_base.values():
            similarity = await self._calculate_similarity(patterns, example)
            if similarity >= threshold:
                example.update_usage()
                similar_examples.append(example)

        return sorted(similar_examples, key=lambda x: x.confidence, reverse=True)

    @abstractmethod
    async def _calculate_similarity(self, patterns: List[str], example: LearningExample) -> float:
        """
        محاسبه میزان شباهت بین الگوها و یک نمونه

        Args:
            patterns: الگوهای استخراج شده
            example: نمونه برای مقایسه

        Returns:
            میزان شباهت (0 تا 1)
        """
        pass

    def _generate_key(self, text: str) -> str:
        """تولید کلید یکتا برای متن"""
        import hashlib
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    async def save_knowledge(self) -> None:
        """ذخیره پایگاه دانش در فایل"""
        try:
            save_data = {
                'knowledge_base': {
                    k: v.to_dict() for k, v in self.knowledge_base.items()
                },
                'patterns': list(self.patterns),
                'timestamp': datetime.now().isoformat()
            }

            async with aiofiles.open(self.save_path / 'knowledge.json', 'w') as f:
                await f.write(json.dumps(save_data, ensure_ascii=False, indent=2))

            self.last_save = datetime.now()
            logger.info("Knowledge base saved successfully")

        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")

    async def load_knowledge(self) -> None:
        """بارگذاری پایگاه دانش از فایل"""
        try:
            file_path = self.save_path / 'knowledge.json'
            if not file_path.exists():
                return

            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                data = json.loads(content)

            self.knowledge_base = {
                k: LearningExample.from_dict(v)
                for k, v in data['knowledge_base'].items()
            }
            self.patterns = set(data['patterns'])
            logger.info(f"Loaded {len(self.knowledge_base)} examples from knowledge base")

        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")

    async def _auto_save(self) -> None:
        """ذخیره خودکار در صورت نیاز"""
        save_interval = self.config.get('save_interval', 3600)  # پیش‌فرض: هر ساعت
        if (datetime.now() - self.last_save).total_seconds() >= save_interval:
            await self.save_knowledge()

    def get_stats(self) -> Dict[str, Any]:
        """دریافت آمار یادگیرنده"""
        verified_count = sum(1 for e in self.knowledge_base.values() if e.verified)
        total_usage = sum(e.usage_count for e in self.knowledge_base.values())

        return {
            'total_examples': len(self.knowledge_base),
            'verified_examples': verified_count,
            'total_patterns': len(self.patterns),
            'total_usage': total_usage,
            'last_save': self.last_save.isoformat()
        }