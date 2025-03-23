# ai/training/data_loader.py
"""
سیستم بارگذاری داده‌های آموزشی

این ماژول مسئول بارگذاری، پیش‌پردازش و آماده‌سازی داده‌ها برای آموزش است.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass

from ..core.common.exceptions import DataLoadingError
from ..models.nlp import SupportedLanguage
from ..models.nlp.base import TextInput


logger = logging.getLogger(__name__)


@dataclass
class DataBatch:
    """نگهدارنده یک دسته از داده‌های آموزشی"""
    inputs: List[TextInput]
    targets: List[Any]
    metadata: Dict[str, Any]
    batch_id: str
    created_at: datetime = datetime.now()


class DatasetStats:
    """آمار و اطلاعات مجموعه داده"""

    def __init__(self):
        self.total_samples: int = 0
        self.samples_per_language: Dict[SupportedLanguage, int] = {}
        self.avg_sequence_length: float = 0.0
        self.last_update: datetime = datetime.now()


class DataLoader:
    """بارگذاری و مدیریت داده‌های آموزشی"""

    def __init__(self, data_dir: str):
        """
        :param data_dir: مسیر دایرکتوری داده‌ها
        """
        self.data_dir = Path(data_dir)
        self.stats = DatasetStats()
        self._cache: Dict[str, DataBatch] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """راه‌اندازی اولیه و اعتبارسنجی مسیرها"""
        try:
            # ایجاد دایرکتوری‌های مورد نیاز
            self.data_dir.mkdir(parents=True, exist_ok=True)
            (self.data_dir / 'raw').mkdir(exist_ok=True)
            (self.data_dir / 'processed').mkdir(exist_ok=True)

            await self._load_stats()
            logger.info("DataLoader initialized successfully")

        except Exception as e:
            logger.error(f"DataLoader initialization failed: {str(e)}")
            raise DataLoadingError("Failed to initialize data loader") from e

    async def _load_stats(self) -> None:
        """بارگذاری آمار داده‌ها"""
        try:
            stats_file = self.data_dir / 'stats.json'
            if stats_file.exists():
                with open(stats_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.stats.total_samples = data.get('total_samples', 0)
                    self.stats.avg_sequence_length = data.get('avg_sequence_length', 0.0)
                    self.stats.last_update = datetime.fromisoformat(data.get('last_update', datetime.now().isoformat()))
        except Exception as e:
            logger.warning(f"Failed to load stats: {str(e)}")
            # در صورت خطا، از مقادیر پیش‌فرض استفاده می‌شود

    async def get_stats(self) -> DatasetStats:
        """دریافت آمار فعلی مجموعه داده"""
        return self.stats

    async def load_batch(self, batch_size: int = 32) -> Optional[DataBatch]:
        """
        بارگذاری یک دسته از داده‌ها

        :param batch_size: اندازه دسته
        :return: دسته داده یا None در صورت عدم وجود داده
        """
        try:
            async with self._lock:
                # بررسی کش
                if cached_batch := self._cache.get(str(batch_size)):
                    return cached_batch

                # بارگذاری از فایل
                processed_dir = self.data_dir / 'processed'
                if not processed_dir.exists():
                    return None

                # خواندن داده‌ها
                for data_file in processed_dir.glob('*.json'):
                    with open(data_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        return await self._create_batch(data, batch_size)

                return None

        except Exception as e:
            logger.error(f"Failed to load batch: {str(e)}")
            raise DataLoadingError("Failed to load data batch") from e

    async def _create_batch(self, data: List[Dict[str, Any]], batch_size: int) -> DataBatch:
        """
        ایجاد یک دسته داده

        :param data: داده‌های خام
        :param batch_size: اندازه دسته
        :return: دسته داده آماده شده
        """
        if not data:
            raise DataLoadingError("Empty data provided")

        # انتخاب داده‌ها به اندازه batch_size
        selected_data = data[:batch_size]

        inputs = []
        targets = []

        for item in selected_data:
            # تبدیل به TextInput
            text_input = TextInput(
                text=item['text'],
                language=SupportedLanguage(item['language']),
                metadata=item.get('metadata', {})
            )
            inputs.append(text_input)

            # استخراج هدف
            targets.append(item['embeddings'])

        return DataBatch(
            inputs=inputs,
            targets=targets,
            metadata={'batch_size': len(inputs)},
            batch_id=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    # ادامه فایل ai/training/data_loader.py

    async def save_data(self, processed_texts: List[TextInput], embeddings: List[Any]) -> None:
        """
        ذخیره داده‌های جدید پردازش شده

        :param processed_texts: لیست متن‌های پردازش شده
        :param embeddings: embedding های متناظر
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.data_dir / 'processed' / f'data_{timestamp}.json'

            # آماده‌سازی داده‌ها برای ذخیره
            data_to_save = []
            for text, embedding in zip(processed_texts, embeddings):
                data_to_save.append({
                    'text': text.text,
                    'language': text.language.value,
                    'metadata': text.metadata,
                    'embeddings': embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                    'timestamp': datetime.now().isoformat()
                })

            # ذخیره داده‌ها
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)

            # بروزرسانی آمار
            await self._update_stats(processed_texts)

            # پاکسازی کش
            await self._clear_cache()

            logger.info(f"Saved {len(processed_texts)} new samples to {output_file}")

        except Exception as e:
            logger.error(f"Failed to save data: {str(e)}")
            raise DataLoadingError("Failed to save processed data") from e

    async def _update_stats(self, new_texts: List[TextInput]) -> None:
        """
        بروزرسانی آمار با داده‌های جدید

        :param new_texts: متن‌های جدید اضافه شده
        """
        try:
            # بروزرسانی تعداد کل نمونه‌ها
            self.stats.total_samples += len(new_texts)

            # بروزرسانی تعداد نمونه‌ها به تفکیک زبان
            for text in new_texts:
                current_count = self.stats.samples_per_language.get(text.language, 0)
                self.stats.samples_per_language[text.language] = current_count + 1

            # بروزرسانی میانگین طول متن‌ها
            text_lengths = [len(text.text.split()) for text in new_texts]
            if text_lengths:
                current_avg = self.stats.avg_sequence_length
                total_samples = self.stats.total_samples
                self.stats.avg_sequence_length = (
                        (current_avg * (total_samples - len(new_texts)) +
                         sum(text_lengths)) / total_samples
                )

            # بروزرسانی زمان آخرین تغییر
            self.stats.last_update = datetime.now()

            # ذخیره آمار
            await self._save_stats()

        except Exception as e:
            logger.error(f"Failed to update stats: {str(e)}")
            # خطا در بروزرسانی آمار نباید مانع ادامه کار شود

    async def _save_stats(self) -> None:
        """ذخیره آمار در فایل"""
        try:
            stats_data = {
                'total_samples': self.stats.total_samples,
                'samples_per_language': {
                    lang.value: count
                    for lang, count in self.stats.samples_per_language.items()
                },
                'avg_sequence_length': self.stats.avg_sequence_length,
                'last_update': self.stats.last_update.isoformat()
            }

            stats_file = self.data_dir / 'stats.json'
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Failed to save stats: {str(e)}")

    async def _clear_cache(self) -> None:
        """پاکسازی کش"""
        async with self._lock:
            self._cache.clear()

    async def preprocess_batch(self, texts: List[str],
                               language: SupportedLanguage) -> List[TextInput]:
        """
        پیش‌پردازش یک دسته از متن‌های خام

        :param texts: لیست متن‌های خام
        :param language: زبان متن‌ها
        :return: لیست TextInput های آماده شده
        """
        processed_texts = []

        for text in texts:
            # حذف فاصله‌های اضافی
            text = ' '.join(text.split())

            # ایجاد TextInput
            text_input = TextInput(
                text=text,
                language=language,
                metadata={
                    'original_length': len(text),
                    'processed_length': len(text.split()),
                    'processed_at': datetime.now().isoformat()
                }
            )
            processed_texts.append(text_input)

        return processed_texts

    async def clean_old_data(self, max_age_days: int = 30) -> None:
        """
        پاکسازی داده‌های قدیمی

        :param max_age_days: حداکثر سن داده‌ها به روز
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            processed_dir = self.data_dir / 'processed'

            for data_file in processed_dir.glob('*.json'):
                # بررسی تاریخ فایل
                file_timestamp = datetime.fromtimestamp(data_file.stat().st_mtime)
                if file_timestamp < cutoff_date:
                    data_file.unlink()
                    logger.info(f"Removed old data file: {data_file}")

            # بروزرسانی آمار پس از پاکسازی
            await self._load_stats()

        except Exception as e:
            logger.error(f"Failed to clean old data: {str(e)}")
            raise DataLoadingError("Failed to clean old data") from e

