"""
پیاده‌سازی پردازشگر متن چندزبانه

این ماژول مسئول پردازش متن در زبان‌های مختلف است. این پیاده‌سازی از ترکیبی از
ابزارهای مختلف برای هر زبان استفاده می‌کند تا بهترین نتیجه را تولید کند.
"""

import logging
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModel
import spacy
from hazm import *
from dataclasses import dataclass
from datetime import datetime

from ...core.common.exceptions import ProcessingError, ValidationError
from ..base import TextPreprocessor, TextInput, ProcessedText, SupportedLanguage

logger = logging.getLogger(__name__)


@dataclass
class LanguageProcessor:
    """نگهدارنده ابزارهای پردازش برای هر زبان"""
    tokenizer: Any
    normalizer: Any
    model: Optional[Any] = None

    async def initialize(self):
        """آماده‌سازی پردازشگر زبان"""
        pass


class MultilingualProcessor(TextPreprocessor):
    """پیاده‌سازی پردازشگر متن چندزبانه"""

    def __init__(self):
        self.processors: Dict[SupportedLanguage, LanguageProcessor] = {}
        self.base_tokenizer = None
        self.base_model = None
        self._initialized = False

    async def initialize(self) -> None:
        """راه‌اندازی اولیه پردازشگرها"""
        try:
            # بارگذاری مدل و توکنایزر پایه
            self.base_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
            self.base_model = AutoModel.from_pretrained('xlm-roberta-base')

            # تنظیم پردازشگر فارسی
            self.processors[SupportedLanguage.PERSIAN] = LanguageProcessor(
                tokenizer=WordTokenizer(),
                normalizer=Normalizer()
            )

            # تنظیم پردازشگر انگلیسی
            english_nlp = spacy.load('en_core_web_sm')
            self.processors[SupportedLanguage.ENGLISH] = LanguageProcessor(
                tokenizer=english_nlp,
                normalizer=english_nlp
            )

            # تنظیم پردازشگر عربی
            # در اینجا می‌توانیم از کتابخانه‌های مخصوص عربی استفاده کنیم
            # فعلاً از مدل پایه استفاده می‌کنیم
            self.processors[SupportedLanguage.ARABIC] = LanguageProcessor(
                tokenizer=self.base_tokenizer,
                normalizer=self.base_tokenizer
            )

            self._initialized = True
            logger.info("Multilingual processor initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize multilingual processor: {str(e)}")
            raise ProcessingError("Initialization failed") from e

    async def preprocess(self, text_input: TextInput) -> ProcessedText:
        """پیش‌پردازش متن ورودی با توجه به زبان آن"""
        if not self._initialized:
            await self.initialize()

        try:
            # اعتبارسنجی ورودی
            if not text_input.validate():
                raise ValidationError("Invalid input text")

            # انتخاب پردازشگر مناسب
            processor = self.processors.get(text_input.language)
            if not processor:
                raise ProcessingError(f"Unsupported language: {text_input.language}")

            # نرمال‌سازی متن
            normalized_text = await self.normalize(text_input.text)

            # توکنایز کردن متن
            tokens = await self.tokenize(normalized_text)

            # ایجاد embedding با استفاده از مدل پایه
            embeddings = await self._create_embeddings(normalized_text)

            # ایجاد خروجی
            return ProcessedText(
                original_input=text_input,
                processed_text=normalized_text,
                tokens=tokens,
                embeddings=embeddings.tolist() if embeddings is not None else None,
                metadata={
                    'language': text_input.language.value,
                    'token_count': len(tokens),
                    'processing_timestamp': datetime.now().isoformat()
                }
            )

        except Exception as e:
            logger.error(f"Text preprocessing failed: {str(e)}")
            raise ProcessingError("Preprocessing failed") from e

    async def normalize(self, text: str) -> str:
        """نرمال‌سازی متن با توجه به قواعد عمومی"""
        # حذف فاصله‌های اضافی
        text = ' '.join(text.split())
        # تبدیل به حروف کوچک (برای انگلیسی)
        text = text.lower()
        # حذف کاراکترهای خاص
        text = ''.join(char for char in text if char.isprintable())
        return text

    async def tokenize(self, text: str) -> List[str]:
        """توکنایز کردن متن با استفاده از توکنایزر پایه"""
        tokens = self.base_tokenizer.tokenize(text)
        return tokens

    async def _create_embeddings(self, text: str) -> torch.Tensor:
        """ایجاد embedding برای متن با استفاده از مدل پایه"""
        try:
            # تبدیل متن به tensor
            inputs = self.base_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            # انتقال به GPU در صورت وجود
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                self.base_model = self.base_model.cuda()

            # محاسبه embedding
            with torch.no_grad():
                outputs = self.base_model(**inputs)
                # استفاده از میانگین آخرین لایه hidden
                embeddings = outputs.last_hidden_state.mean(dim=1)

            return embeddings.cpu()

        except Exception as e:
            logger.error(f"Failed to create embeddings: {str(e)}")
            return None

    async def cleanup(self):
        """آزادسازی منابع"""
        # آزادسازی مدل‌ها از حافظه
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.base_model = None
        self.base_tokenizer = None
        self._initialized = False
        logger.info("Multilingual processor cleaned up")