import os
import json
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from pathlib import Path
import aiofiles
import magic  # برای تشخیص نوع فایل
import chardet  # برای تشخیص کدگذاری متن

from ..interfaces.knowledge_source import (
    KnowledgeSource,
    KnowledgeSourceType,
    KnowledgeMetadata,
    KnowledgePriority,
    LearningContext
)

logger = logging.getLogger(__name__)


class FileTypes:
    """انواع فایل‌های پشتیبانی شده"""
    TEXT = ['text/plain', 'text/csv', 'text/markdown']
    JSON = ['application/json']
    PDF = ['application/pdf']
    OFFICE = ['application/msword', 'application/vnd.openxmlformats-officedocument']


class LocalFileSource(KnowledgeSource):
    def __init__(self, source_config: Dict[str, Any]):
        super().__init__(source_config)
        self.base_path = Path(source_config.get('base_path', '.'))
        self.supported_extensions = source_config.get('supported_extensions',
                                                      ['.txt', '.md', '.json', '.csv', '.pdf', '.doc', '.docx'])
        self.max_file_size = source_config.get('max_file_size', 10 * 1024 * 1024)  # 10MB
        self._file_handlers = {}

    async def initialize(self) -> bool:
        try:
            if not self.base_path.exists():
                self.base_path.mkdir(parents=True)

            self._file_handlers = {
                '.txt': self._handle_text_file,
                '.md': self._handle_text_file,
                '.json': self._handle_json_file,
                '.csv': self._handle_csv_file,
                '.pdf': self._handle_pdf_file
            }

            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize local file source: {str(e)}")
            return False

    async def get_knowledge(self, file_path: str,
                            context: Optional[LearningContext] = None) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("Local file source is not initialized")

        full_path = self.base_path / file_path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = full_path.stat().st_size
        if file_size > self.max_file_size:
            raise ValueError(f"File size exceeds maximum limit: {file_size} > {self.max_file_size}")

        try:
            file_type = magic.from_file(str(full_path), mime=True)
            extension = full_path.suffix.lower()

            if extension not in self.supported_extensions:
                raise ValueError(f"Unsupported file type: {extension}")

            handler = self._file_handlers.get(extension)
            if not handler:
                raise ValueError(f"No handler for file type: {extension}")

            async with aiofiles.open(full_path, 'rb') as file:
                content = await file.read()

            processed_content = await handler(content, context)

            metadata = await self._create_metadata(full_path, content, processed_content)

            return {
                'content': processed_content,
                'metadata': metadata,
                'file_info': {
                    'path': str(full_path),
                    'size': file_size,
                    'type': file_type
                }
            }

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

    async def _handle_text_file(self, content: bytes,
                                context: Optional[LearningContext]) -> str:
        """پردازش فایل‌های متنی"""
        encoding = chardet.detect(content)['encoding'] or 'utf-8'
        text = content.decode(encoding)

        # حذف خطوط خالی اضافی و نرمال‌سازی فضای خالی
        lines = [line.strip() for line in text.splitlines()]
        return '\n'.join(line for line in lines if line)

    async def _handle_json_file(self, content: bytes,
                                context: Optional[LearningContext]) -> Dict:
        """پردازش فایل‌های JSON"""
        encoding = chardet.detect(content)['encoding'] or 'utf-8'
        text = content.decode(encoding)
        return json.loads(text)

    async def _handle_csv_file(self, content: bytes,
                               context: Optional[LearningContext]) -> List[Dict]:
        """پردازش فایل‌های CSV"""
        import pandas as pd
        import io

        df = pd.read_csv(io.BytesIO(content))
        return df.to_dict('records')

    async def _handle_pdf_file(self, content: bytes,
                               context: Optional[LearningContext]) -> str:
        """پردازش فایل‌های PDF"""
        # اینجا می‌توانیم از کتابخانه‌های مختلف PDF استفاده کنیم
        # فعلاً یک نسخه ساده پیاده‌سازی می‌کنیم
        import pdfplumber

        text_content = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                text_content.append(page.extract_text())

        return '\n'.join(text_content)

    async def _create_metadata(self, file_path: Path, raw_content: bytes,
                               processed_content: Any) -> KnowledgeMetadata:
        """ایجاد متادیتا برای محتوای فایل"""
        file_stats = file_path.stat()

        return KnowledgeMetadata(
            source_id=str(file_path),
            source_type=KnowledgeSourceType.LOCAL_FILE,
            created_at=datetime.fromtimestamp(file_stats.st_ctime),
            updated_at=datetime.fromtimestamp(file_stats.st_mtime),
            priority=self._determine_priority(file_path),
            confidence_score=1.0,  # فایل‌های محلی قابل اعتماد هستند
            validation_status=True,
            learning_progress=0.0,
            tags=self._extract_tags(file_path, processed_content),
            language=self._detect_language(processed_content),
            size_bytes=len(raw_content),
            checksum=hashlib.sha256(raw_content).hexdigest()
        )

    def _determine_priority(self, file_path: Path) -> KnowledgePriority:
        """تعیین اولویت بر اساس نام و مسیر فایل"""
        name = file_path.stem.lower()
        if any(kw in name for kw in ['important', 'critical', 'urgent', 'مهم', 'ضروری']):
            return KnowledgePriority.HIGH
        return KnowledgePriority.MEDIUM

    def _extract_tags(self, file_path: Path, content: Any) -> List[str]:
        """استخراج برچسب‌ها از نام فایل و محتوا"""
        tags = [file_path.suffix[1:]]  # نوع فایل به عنوان اولین برچسب

        # افزودن نام پوشه به برچسب‌ها
        parent_folders = file_path.parent.parts
        tags.extend(folder for folder in parent_folders if folder != '.')

        return list(set(tags))

    def _detect_language(self, content: Any) -> str:
        """تشخیص زبان محتوا"""
        if isinstance(content, str):
            # تشخیص ساده بر اساس کاراکترها
            if any('\u0600' <= ch <= '\u06FF' for ch in content):
                return 'fa'
            if any('\u0750' <= ch <= '\u077F' for ch in content):
                return 'ar'
        return 'en'

    async def validate_knowledge(self, knowledge_data: Dict[str, Any]) -> bool:
        """اعتبارسنجی دانش استخراج شده از فایل"""
        try:
            if not knowledge_data.get('content'):
                return False

            file_info = knowledge_data.get('file_info', {})
            if not file_info.get('path') or not file_info.get('type'):
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating file knowledge: {str(e)}")
            return False

    async def update_learning_progress(self, knowledge_id: str, progress: float) -> None:
        """به‌روزرسانی پیشرفت یادگیری"""
        if not 0 <= progress <= 1:
            raise ValueError("Progress must be between 0 and 1")

        logger.info(f"Learning progress for file {knowledge_id}: {progress}")

    async def cleanup(self) -> None:
        """پاکسازی منابع"""
        self._initialized = False
        logger.info("Local file source cleaned up")