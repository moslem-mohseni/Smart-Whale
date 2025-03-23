# ai/training/pipeline.py
"""
سیستم آموزش مدل‌های هوش مصنوعی

این ماژول مسئول مدیریت فرآیند آموزش مدل‌هاست. این سیستم به صورت مستمر داده‌های جدید
را جمع‌آوری می‌کند، کیفیت آنها را بررسی می‌کند، و در زمان مناسب آموزش مجدد را انجام می‌دهد.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import asyncio
from dataclasses import dataclass
from pathlib import Path

from ..core.common.exceptions import TrainingError
from ..models.nlp.base import ProcessedText, TextInput


logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """تنظیمات آموزش"""
    batch_size: int = 32
    learning_rate: float = 1e-5
    num_epochs: int = 10
    validation_split: float = 0.2
    early_stopping_patience: int = 3
    min_samples_required: int = 1000
    max_samples_per_class: int = 10000
    data_collection_interval: int = 3600  # یک ساعت

@dataclass
class TrainingMetrics:
    """معیارهای ارزیابی آموزش"""
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    timestamp: datetime = datetime.now()

class DataCollector:
    """جمع‌آوری و مدیریت داده‌های آموزشی"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._current_batch: List[Dict[str, Any]] = []
        self._batch_counter = 0

    async def add_sample(self, input_data: TextInput, result: Dict[str, Any],
                        gpt_result: Optional[Dict[str, Any]] = None):
        """افزودن یک نمونه جدید به مجموعه داده"""
        try:
            # ساخت نمونه آموزشی
            training_sample = {
                'input': {
                    'text': input_data.text,
                    'language': input_data.language.value,
                    'metadata': input_data.metadata
                },
                'model_result': result,
                'gpt_result': gpt_result,
                'timestamp': datetime.now().isoformat()
            }

            # افزودن به بچ فعلی
            self._current_batch.append(training_sample)

            # ذخیره بچ در صورت پر شدن
            if len(self._current_batch) >= 100:  # اندازه بچ
                await self._save_batch()

        except Exception as e:
            logger.error(f"Failed to add training sample: {str(e)}")
            raise TrainingError("Data collection failed") from e

    async def _save_batch(self):
        """ذخیره یک بچ از داده‌ها"""
        if not self._current_batch:
            return

        try:
            batch_file = self.data_dir / f"batch_{self._batch_counter}.json"
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(self._current_batch, f, ensure_ascii=False, indent=2)

            self._batch_counter += 1
            self._current_batch = []
            logger.info(f"Saved training batch to {batch_file}")

        except Exception as e:
            logger.error(f"Failed to save batch: {str(e)}")
            raise TrainingError("Batch saving failed") from e

class TrainingPipeline:
    """خط لوله آموزش مدل"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_collector = DataCollector(Path("data/training"))
        self._training_task: Optional[asyncio.Task] = None
        self._should_stop = False
        self.current_metrics: Optional[TrainingMetrics] = None

    async def start(self):
        """شروع فرآیند آموزش مستمر"""
        if self._training_task is None:
            self._should_stop = False
            self._training_task = asyncio.create_task(self._continuous_training())
            logger.info("Training pipeline started")

    async def stop(self):
        """توقف فرآیند آموزش"""
        if self._training_task:
            self._should_stop = True
            await self._training_task
            self._training_task = None
            logger.info("Training pipeline stopped")

    async def _continuous_training(self):
        """فرآیند مستمر آموزش"""
        while not self._should_stop:
            try:
                # بررسی نیاز به آموزش
                if await self._should_train():
                    await self._perform_training()

                # انتظار تا دور بعدی
                await asyncio.sleep(self.config.data_collection_interval)

            except Exception as e:
                logger.error(f"Training cycle failed: {str(e)}")
                await asyncio.sleep(60)  # انتظار کوتاه قبل از تلاش مجدد

    async def _should_train(self) -> bool:
        """تصمیم‌گیری برای شروع آموزش"""
        try:
            # بررسی تعداد نمونه‌های جدید
            new_samples_count = await self._count_new_samples()
            if new_samples_count < self.config.min_samples_required:
                return False

            # بررسی کیفیت عملکرد فعلی
            if self.current_metrics:
                if self.current_metrics.accuracy > 0.95:
                    # اگر عملکرد خیلی خوب است، نیازی به آموزش نیست
                    return False

            return True

        except Exception as e:
            logger.error(f"Training decision failed: {str(e)}")
            return False

    async def _perform_training(self):
        """انجام فرآیند آموزش"""
        try:
            logger.info("Starting training cycle")

            # بارگذاری داده‌ها
            training_data = await self._load_training_data()
            if not training_data:
                return

            # تقسیم داده‌ها به آموزش و اعتبارسنجی
            train_data, val_data = self._split_data(training_data)

            # آموزش مدل
            metrics = await self._train_model(train_data, val_data)

            # بروزرسانی متریک‌ها
            self.current_metrics = metrics

            logger.info(f"Training completed with metrics: {metrics}")

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise TrainingError("Training cycle failed") from e

    async def _count_new_samples(self) -> int:
        """شمارش نمونه‌های جدید از آخرین آموزش"""
        try:
            count = 0
            batch_files = list((self.data_collector.data_dir / "processed").glob("*.json"))

            for batch_file in batch_files:
                if batch_file.stat().st_mtime > self.last_training_time:
                    with open(batch_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        count += len(data)

            return count
        except Exception as e:
            logger.error(f"Failed to count new samples: {e}")
            return 0

    async def _load_training_data(self) -> List[Dict[str, Any]]:
        """بارگذاری داده‌های آموزشی"""
        try:
            all_data = []
            batch_files = list((self.data_collector.data_dir / "processed").glob("*.json"))

            for batch_file in batch_files:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if self._validate_batch(data):
                        all_data.extend(data)

            if len(all_data) > self.config.max_samples_per_class:
                all_data = self._balance_data(all_data)

            return all_data
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return []

    def _validate_batch(self, batch_data: List[Dict[str, Any]]) -> bool:
        """اعتبارسنجی داده‌های یک بچ"""
        required_fields = {'input', 'model_result', 'timestamp'}
        return all(all(field in item for field in required_fields) for item in batch_data)

    def _balance_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """متوازن‌سازی داده‌ها"""
        return sorted(data, key=lambda x: x['timestamp'])[-self.config.max_samples_per_class:]

    def _split_data(self, data: List[Dict[str, Any]]) -> tuple:
        """تقسیم داده‌ها به مجموعه آموزش و اعتبارسنجی"""
        split_idx = int(len(data) * (1 - self.config.validation_split))
        return data[:split_idx], data[split_idx:]

    async def _train_model(self, train_data: List[Dict[str, Any]],
                           val_data: List[Dict[str, Any]]) -> TrainingMetrics:
        """آموزش مدل با داده‌های جدید"""
        try:
            current_epoch = 0
            best_loss = float('inf')
            patience_counter = 0

            while current_epoch < self.config.num_epochs:
                # آموزش یک اپاک
                train_metrics = await self._train_epoch(train_data)
                val_metrics = await self._validate_epoch(val_data)

                # بررسی early stopping
                if val_metrics.loss < best_loss:
                    best_loss = val_metrics.loss
                    patience_counter = 0
                    await self._save_model_checkpoint()
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info("Early stopping triggered")
                        break

                current_epoch += 1

            return val_metrics

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise TrainingError("Training failed") from e

