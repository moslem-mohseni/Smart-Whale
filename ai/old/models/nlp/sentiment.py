# ai/models/nlp/sentiment.py
"""
پیاده‌سازی تحلیل‌گر احساسات متن

این ماژول مسئول تشخیص و تحلیل احساسات در متون مختلف است. از یک مدل چندزبانه
برای تحلیل احساسات استفاده می‌کند و قابلیت کار با زبان‌های مختلف را دارد.

نکته معماری: این کلاس از TextAnalyzer ارث‌بری می‌کند و با MultilingualProcessor
همکاری می‌کند تا اصل Single Responsibility را رعایت کند.
"""

import logging
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from datetime import datetime

from ..base import TextAnalyzer, ProcessedText, SupportedLanguage
from ...core.common.exceptions import AnalysisError
from .multilingual import MultilingualProcessor

logger = logging.getLogger(__name__)


class EmotionLabel:
    """کلاس ثابت‌های برچسب‌های احساسی"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class SentimentAnalyzer(TextAnalyzer):
    """تحلیل‌گر احساسات متن با پشتیبانی از چند زبان"""

    def __init__(self, multilingual_processor: MultilingualProcessor):
        """
        مقداردهی اولیه تحلیل‌گر احساسات

        Args:
            multilingual_processor: نمونه‌ای از پردازشگر چندزبانه که قبلاً ایجاد شده
        """
        self.multilingual_processor = multilingual_processor
        self.model = None
        self.tokenizer = None
        self._initialized = False
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

    async def initialize(self) -> None:
        """راه‌اندازی اولیه مدل تحلیل احساسات"""
        try:
            # بارگذاری مدل پایه تحلیل احساسات چندزبانه
            model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

            # انتقال مدل به GPU در صورت وجود
            self.model = self.model.to(self._device)
            self.model.eval()  # تنظیم مدل در حالت ارزیابی

            self._initialized = True
            logger.info(f"Sentiment analyzer initialized on {self._device}")

        except Exception as e:
            logger.error(f"Failed to initialize sentiment analyzer: {str(e)}")
            raise AnalysisError("Sentiment analyzer initialization failed") from e

    async def analyze_sentiment(self, processed_text: ProcessedText) -> Dict[str, float]:
        """
        تحلیل احساسات متن پردازش شده

        Args:
            processed_text: متن پردازش شده توسط پردازشگر چندزبانه

        Returns:
            دیکشنری شامل احتمال هر نوع احساس
        """
        if not self._initialized:
            await self.initialize()

        try:
            # آماده‌سازی ورودی برای مدل
            inputs = self.tokenizer(
                processed_text.processed_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self._device)

            # پیش‌بینی احساسات
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)
                scores = scores.cpu().numpy()[0]

            # تبدیل نتایج به دیکشنری
            sentiment_scores = {
                EmotionLabel.NEGATIVE: float(scores[0]),
                EmotionLabel.NEUTRAL: float(scores[2]),
                EmotionLabel.POSITIVE: float(scores[4])
            }

            # افزودن برچسب احساس غالب
            dominant_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])
            if dominant_sentiment[1] < 0.5:
                sentiment_scores[EmotionLabel.MIXED] = 1.0
                primary_emotion = EmotionLabel.MIXED
            else:
                primary_emotion = dominant_sentiment[0]

            # افزودن متادیتا به نتایج
            sentiment_scores.update({
                'primary_emotion': primary_emotion,
                'confidence': float(dominant_sentiment[1]),
                'language': processed_text.original_input.language.value,
                'analysis_timestamp': datetime.now().isoformat()
            })

            logger.debug(f"Sentiment analysis completed for text in {processed_text.original_input.language}")
            return sentiment_scores

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            raise AnalysisError("Failed to analyze sentiment") from e

    async def extract_entities(self, processed_text: ProcessedText) -> List[Dict[str, Any]]:
        """
        استخراج موجودیت‌های مرتبط با احساسات

        این متد می‌تواند عبارات و کلمات کلیدی که بار احساسی دارند را شناسایی کند.
        """
        try:
            # شناسایی عبارات با بار احساسی با استفاده از توکن‌ها
            emotional_entities = []
            for i, token in enumerate(processed_text.tokens):
                # محاسبه احساس برای هر توکن
                token_sentiment = await self._analyze_token_sentiment(token)
                if token_sentiment['intensity'] > 0.5:  # فقط موارد قوی
                    emotional_entities.append({
                        'text': token,
                        'position': i,
                        'sentiment': token_sentiment['sentiment'],
                        'intensity': token_sentiment['intensity']
                    })

            return emotional_entities

        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            return []

    async def classify_text(self, processed_text: ProcessedText) -> Dict[str, float]:
        """
        دسته‌بندی متن بر اساس نوع احساس

        این متد یک لایه بالاتر از تحلیل احساسات ساده است و می‌تواند
        دسته‌بندی‌های دقیق‌تری ارائه دهد.
        """
        try:
            # تحلیل احساسات پایه
            base_sentiment = await self.analyze_sentiment(processed_text)

            # افزودن دسته‌بندی‌های پیشرفته‌تر
            classifications = {
                'emotional_intensity': self._calculate_emotional_intensity(base_sentiment),
                'sentiment_stability': self._calculate_sentiment_stability(base_sentiment),
                'sentiment_complexity': self._calculate_sentiment_complexity(base_sentiment)
            }

            return {**base_sentiment, **classifications}

        except Exception as e:
            logger.error(f"Text classification failed: {str(e)}")
            raise AnalysisError("Classification failed") from e

    async def _analyze_token_sentiment(self, token: str) -> Dict[str, float]:
        """تحلیل احساس یک توکن منفرد"""
        # پیاده‌سازی ساده برای نمونه
        return {
            'sentiment': 'neutral',
            'intensity': 0.0
        }

    def _calculate_emotional_intensity(self, sentiment_scores: Dict[str, float]) -> float:
        """محاسبه شدت احساسی متن"""
        return abs(sentiment_scores[EmotionLabel.POSITIVE] - sentiment_scores[EmotionLabel.NEGATIVE])

    def _calculate_sentiment_stability(self, sentiment_scores: Dict[str, float]) -> float:
        """محاسبه ثبات احساسی متن"""
        return 1.0 - sentiment_scores.get(EmotionLabel.MIXED, 0.0)

    def _calculate_sentiment_complexity(self, sentiment_scores: Dict[str, float]) -> float:
        """محاسبه پیچیدگی احساسی متن"""
        scores = [v for k, v in sentiment_scores.items() if k in
                  [EmotionLabel.POSITIVE, EmotionLabel.NEGATIVE, EmotionLabel.NEUTRAL]]
        return -sum(s * np.log2(s) for s in scores if s > 0)  # آنتروپی شانون

    async def cleanup(self):
        """آزادسازی منابع"""
        if self.model is not None:
            self.model = self.model.cpu()
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.tokenizer = None
        self._initialized = False
        logger.info("Sentiment analyzer cleaned up")