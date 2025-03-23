import aiohttp
import asyncio
import json
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from ..interfaces.knowledge_source import (
    KnowledgeSource,
    KnowledgeSourceType,
    KnowledgeMetadata,
    KnowledgePriority,
    LearningContext
)

logger = logging.getLogger(__name__)


class ChatGPTSource(KnowledgeSource):
    """پیاده‌سازی منبع دانش ChatGPT"""

    def __init__(self, source_config: Dict[str, Any]):
        super().__init__(source_config)
        self.api_key = source_config.get('api_key')
        self.api_base = source_config.get('api_base', 'https://api.openai.com/v1')
        self.model = source_config.get('model', 'gpt-3.5-turbo')
        self.session = None
        self.rate_limit = source_config.get('rate_limit', {
            'requests_per_minute': 60,
            'tokens_per_minute': 90000
        })
        self._request_times = []
        self._token_usage = []

    async def initialize(self) -> bool:
        """راه‌اندازی اتصال به ChatGPT"""
        try:
            if not self.api_key:
                raise ValueError("API key is required for ChatGPT")

            self.session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            self._initialized = True
            logger.info("ChatGPT source initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ChatGPT source: {str(e)}")
            return False

    async def _check_rate_limits(self) -> bool:
        """بررسی محدودیت‌های نرخ درخواست"""
        current_time = datetime.now()

        # پاکسازی درخواست‌های قدیمی
        self._request_times = [t for t in self._request_times
                               if (current_time - t).seconds < 60]
        self._token_usage = [(t, count) for t, count in self._token_usage
                             if (current_time - t).seconds < 60]

        # بررسی محدودیت‌ها
        if len(self._request_times) >= self.rate_limit['requests_per_minute']:
            logger.warning("Rate limit exceeded for requests per minute")
            return False

        total_tokens = sum(count for _, count in self._token_usage)
        if total_tokens >= self.rate_limit['tokens_per_minute']:
            logger.warning("Rate limit exceeded for tokens per minute")
            return False

        return True

    def _update_rate_limits(self, token_count: int) -> None:
        """به‌روزرسانی شمارنده‌های نرخ درخواست"""
        current_time = datetime.now()
        self._request_times.append(current_time)
        self._token_usage.append((current_time, token_count))

    async def get_knowledge(self, query: str,
                            context: Optional[LearningContext] = None) -> Dict[str, Any]:
        """
        دریافت دانش از ChatGPT

        Args:
            query: پرس‌وجوی دانش
            context: زمینه یادگیری

        Returns:
            دیکشنری حاوی دانش و متادیتا
        """
        if not self._initialized or not self.session:
            raise RuntimeError("ChatGPT source is not initialized")

        try:
            # بررسی محدودیت نرخ
            if not await self._check_rate_limits():
                await asyncio.sleep(5)  # انتظار قبل از تلاش مجدد

            # ساخت پیام با در نظر گرفتن context
            messages = self._build_messages(query, context)

            async with self.session.post(
                    f"{self.api_base}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0.7
                    }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error: {error_text}")

                data = await response.json()
                response_text = data['choices'][0]['message']['content']
                token_count = data['usage']['total_tokens']

                # به‌روزرسانی شمارنده‌ها
                self._update_rate_limits(token_count)

                # ساخت متادیتا
                metadata = KnowledgeMetadata(
                    source_id=self.config['source_id'],
                    source_type=KnowledgeSourceType.CHAT_GPT,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    priority=KnowledgePriority.HIGH,
                    confidence_score=0.85,  # مقدار پیش‌فرض برای ChatGPT
                    validation_status=True,
                    learning_progress=0.0,
                    tags=self._extract_tags(response_text),
                    language=self._detect_language(response_text),
                    size_bytes=len(response_text.encode('utf-8')),
                    checksum=hashlib.sha256(response_text.encode()).hexdigest()
                )

                return {
                    'content': response_text,
                    'metadata': metadata,
                    'raw_response': data
                }

        except Exception as e:
            logger.error(f"Error getting knowledge from ChatGPT: {str(e)}")
            raise

    def _build_messages(self, query: str, context: Optional[LearningContext]) -> list:
        """ساخت پیام‌ها برای درخواست به ChatGPT"""
        messages = [{"role": "user", "content": query}]

        if context:
            system_message = (
                f"Focus on: {', '.join(context.focus_areas)}. "
                f"Mode: {context.learning_mode}. "
                f"Required confidence: {context.required_confidence}"
            )
            messages.insert(0, {"role": "system", "content": system_message})

        return messages

    def _extract_tags(self, content: str) -> List[str]:
        """استخراج برچسب‌های موضوعی از محتوا"""
        # در نسخه‌های بعدی پیاده‌سازی می‌شود
        return []

    def _detect_language(self, content: str) -> str:
        """تشخیص زبان محتوا"""
        # در نسخه‌های بعدی پیاده‌سازی می‌شود
        return 'fa' if any('\u0600' <= ch <= '\u06FF' for ch in content) else 'en'

    async def validate_knowledge(self, knowledge_data: Dict[str, Any]) -> bool:
        """اعتبارسنجی دانش دریافتی"""
        try:
            content = knowledge_data.get('content', '')
            # بررسی‌های اولیه
            if not content or len(content) < 10:
                return False

            # در نسخه‌های بعدی، اعتبارسنجی‌های پیشرفته‌تر اضافه می‌شود
            return True

        except Exception as e:
            logger.error(f"Error validating knowledge: {str(e)}")
            return False

    async def update_learning_progress(self, knowledge_id: str, progress: float) -> None:
        """به‌روزرسانی پیشرفت یادگیری"""
        if not 0 <= progress <= 1:
            raise ValueError("Progress must be between 0 and 1")

        # در نسخه‌های بعدی پیاده‌سازی می‌شود
        logger.info(f"Learning progress for {knowledge_id}: {progress}")

    async def cleanup(self) -> None:
        """پاکسازی منابع"""
        if self.session:
            await self.session.close()
            self.session = None
            self._initialized = False
            logger.info("ChatGPT source cleaned up")