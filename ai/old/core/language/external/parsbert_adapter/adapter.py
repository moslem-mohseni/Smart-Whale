import torch
import logging
import asyncio
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from functools import lru_cache

from .config import AdapterConfig
from .errors import ModelNotInitializedError, ProcessingError, InvalidInputError
from .retry import retry_operation
from .utils import manage_memory, check_gpu_status

logger = logging.getLogger(__name__)


class ParsBERTAdapter:
    def __init__(self, model_name: str = "HooshvareLab/bert-fa-base-uncased", config: Optional[AdapterConfig] = None):
        self.config = config or AdapterConfig()
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing ParsBERT adapter (device: {self.device})")
        self._adjust_batch_size()

    async def initialize(self) -> bool:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self._adjust_batch_size()
            logger.info("ParsBERT model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ParsBERT: {e}")
            return False

    def validate_model(self):
        if not self.model or not self.tokenizer:
            raise ModelNotInitializedError("ParsBERT model is not initialized. Please call initialize() first.")

    def _adjust_batch_size(self):
        """تنظیم خودکار batch_size بر اساس میزان حافظه GPU"""
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            if total_memory < 8:
                self.config.batch_size = min(4, self.config.batch_size)
            elif total_memory < 16:
                self.config.batch_size = min(8, self.config.batch_size)
            else:
                self.config.batch_size = min(16, self.config.batch_size)
            logger.info(f"Batch size adjusted to {self.config.batch_size} based on GPU memory")

    @retry_operation()
    @lru_cache(maxsize=1000)
    async def process_text(self, text: str) -> torch.Tensor:
        if not text.strip():
            raise InvalidInputError("Empty text provided")
        self.validate_model()

        try:
            encoded = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True,
                                     max_length=self.config.max_length).to(self.device)
            with torch.no_grad():
                outputs = self.model(**encoded)
                return outputs.last_hidden_state.mean(dim=1)
        except torch.cuda.OutOfMemoryError:
            manage_memory()
            self._adjust_batch_size()
            raise ProcessingError("GPU memory overloaded")
        except Exception as e:
            logger.error(f"Unexpected error in process_text: {e}")
            raise ProcessingError(str(e))

    async def prefetch_texts(self, texts: List[str]) -> None:
        """اجرای پیش‌پردازش برای متون آینده جهت بهینه‌سازی سرعت"""
        logger.info("Prefetching texts for faster inference.")
        for text in texts:
            asyncio.create_task(self.process_text(text))

    async def cleanup(self):
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        torch.cuda.empty_cache()
        logger.info("ParsBERT resources cleaned up")

    def status(self) -> Dict[str, Any]:
        check_gpu_status()
        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.config.batch_size,
            "is_initialized": self.model is not None
        }
