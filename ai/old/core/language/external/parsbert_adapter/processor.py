import torch
import asyncio
import logging
from typing import List, Optional
from .errors import ModelNotInitializedError, ProcessingError
from .utils import manage_memory, estimate_available_memory
from .adapter import ParsBERTAdapter

logger = logging.getLogger(__name__)


class ParsBERTProcessor:
    def __init__(self, adapter: ParsBERTAdapter):
        self.adapter = adapter
        self.batch_size = adapter.config.batch_size

    async def batch_process(self, texts: List[str]) -> List[Optional[torch.Tensor]]:
        """پردازش گروهی متون با مدیریت کارایی و تنظیم خودکار batch_size"""
        if not self.adapter.model or not self.adapter.tokenizer:
            raise ModelNotInitializedError("ParsBERT model is not initialized. Please call initialize() first.")

        available_memory = estimate_available_memory()
        if available_memory and available_memory < 2.0:
            self.batch_size = max(4, self.batch_size // 2)
            logger.warning(f"Reducing batch size to {self.batch_size} due to low available GPU memory")

        results = []
        stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                with torch.cuda.stream(stream) if stream else torch.no_grad():
                    encoded = self.adapter.tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(
                        self.adapter.device)
                    outputs = self.adapter.model(**encoded)
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                    results.extend(batch_embeddings.cpu())
            except torch.cuda.OutOfMemoryError:
                manage_memory()
                self.batch_size = max(4, self.batch_size // 2)
                logger.warning(f"Batch size reduced to {self.batch_size} due to OOM error.")
                raise ProcessingError("GPU memory overloaded during batch processing")
            except Exception as e:
                logger.error(f"Unexpected error in batch processing: {e}")
                results.extend([None] * len(batch))

        return results
