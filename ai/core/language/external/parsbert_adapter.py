"""
ParsBERT Adapter Module
---------------------
این ماژول یک لایه انتزاع برای کار با مدل ParsBERT فراهم می‌کند. هدف این adapter جداسازی
پیچیدگی‌های ParsBERT از سیستم اصلی و فراهم کردن یک رابط ساده و یکپارچه است.
"""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ParsBERTResult:
    """نتایج پردازش متن توسط ParsBERT"""
    embeddings: torch.Tensor  # بردارهای معنایی کلمات
    sentence_embedding: torch.Tensor  # بردار معنایی کل جمله
    tokens: List[str]  # توکن‌های متن
    attention_weights: torch.Tensor  # وزن‌های توجه
    timestamp: datetime  # زمان پردازش


class ParsBERTAdapter:
    """
    آداپتور برای کار با مدل ParsBERT

    این کلاس مسئول مدیریت ارتباط با مدل ParsBERT است و یک رابط ساده برای
    استفاده از قابلیت‌های آن فراهم می‌کند.
    """

    def __init__(self, model_name: str = "HooshvareLab/bert-fa-base-uncased"):
        """
        مقداردهی اولیه آداپتور ParsBERT

        Args:
            model_name: نام یا مسیر مدل ParsBERT
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing ParsBERT adapter (device: {self.device})")

    async def initialize(self) -> bool:
        """
        راه‌اندازی و بارگیری مدل

        Returns:
            موفقیت در راه‌اندازی
        """
        try:
            # بارگیری توکنایزر و مدل
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)

            # انتقال مدل به GPU در صورت وجود
            self.model = self.model.to(self.device)
            self.model.eval()  # تنظیم مدل در حالت ارزیابی

            logger.info("ParsBERT model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ParsBERT: {e}")
            return False

    async def process_text(self, text: str) -> Optional[ParsBERTResult]:
        """
        پردازش متن با استفاده از ParsBERT

        Args:
            text: متن فارسی برای پردازش

        Returns:
            نتایج پردازش یا None در صورت خطا
        """
        try:
            if not self.model or not self.tokenizer:
                raise RuntimeError("ParsBERT is not initialized")

            # توکن‌سازی متن
            encoded = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )

            # انتقال داده‌ها به دستگاه مناسب
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # اجرای مدل
            with torch.no_grad():
                outputs = self.model(**encoded)

                # استخراج نتایج
                embeddings = outputs.last_hidden_state
                attention = outputs.attentions[-1] if outputs.attentions else None
                tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])

                # محاسبه embedding کل جمله با میانگین‌گیری
                sentence_embedding = embeddings.mean(dim=1)

                result = ParsBERTResult(
                    embeddings=embeddings.cpu(),  # انتقال به CPU برای استفاده عمومی
                    sentence_embedding=sentence_embedding.cpu(),
                    tokens=tokens,
                    attention_weights=attention.cpu() if attention is not None else None,
                    timestamp=datetime.now()
                )

                return result

        except Exception as e:
            logger.error(f"Error processing text with ParsBERT: {e}")
            return None

    async def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        محاسبه میزان شباهت معنایی بین دو متن

        Args:
            text1: متن اول
            text2: متن دوم

        Returns:
            میزان شباهت بین 0 تا 1
        """
        try:
            result1 = await self.process_text(text1)
            result2 = await self.process_text(text2)

            if result1 and result2:
                # محاسبه شباهت کسینوسی بین بردارهای جمله
                similarity = torch.nn.functional.cosine_similarity(
                    result1.sentence_embedding,
                    result2.sentence_embedding
                )
                return float(similarity.item())

            return 0.0

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def get_config(self) -> Dict[str, Any]:
        """دریافت تنظیمات فعلی آداپتور"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'is_initialized': self.model is not None,
            'max_length': self.tokenizer.model_max_length if self.tokenizer else None
        }