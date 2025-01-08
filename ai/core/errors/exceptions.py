from typing import Optional, Any, Dict
from datetime import datetime

class AIError(Exception):
    """کلاس پایه برای تمام خطاهای سیستم هوش مصنوعی"""
    def __init__(self, message: str, error_code: str = None, details: Dict = None):
        self.message = message
        self.error_code = error_code or 'AI_ERROR'
        self.details = details or {}
        self.timestamp = datetime.now()
        super().__init__(self.message)

class KnowledgeSourceError(AIError):
    """خطاهای مربوط به منابع دانش"""
    def __init__(self, message: str, source_id: str, **kwargs):
        super().__init__(
            message=message,
            error_code='KNOWLEDGE_SOURCE_ERROR',
            details={'source_id': source_id, **kwargs}
        )

class InitializationError(AIError):
    """خطای راه‌اندازی"""
    def __init__(self, message: str, component: str, **kwargs):
        super().__init__(
            message=message,
            error_code='INIT_ERROR',
            details={'component': component, **kwargs}
        )

class ValidationError(AIError):
    """خطای اعتبارسنجی"""
    def __init__(self, message: str, validation_errors: Dict[str, str], **kwargs):
        super().__init__(
            message=message,
            error_code='VALIDATION_ERROR',
            details={'errors': validation_errors, **kwargs}
        )

class LearningError(AIError):
    """خطای یادگیری"""
    def __init__(self, message: str, query: str, **kwargs):
        super().__init__(
            message=message,
            error_code='LEARNING_ERROR',
            details={'query': query, **kwargs}
        )

class ResourceError(AIError):
    """خطای منابع سیستم"""
    def __init__(self, message: str, resource_type: str, **kwargs):
        super().__init__(
            message=message,
            error_code='RESOURCE_ERROR',
            details={'resource_type': resource_type, **kwargs}
        )

class APIError(AIError):
    """خطای ارتباط با APIهای خارجی"""
    def __init__(self, message: str, api_name: str, status_code: Optional[int] = None, **kwargs):
        super().__init__(
            message=message,
            error_code='API_ERROR',
            details={
                'api_name': api_name,
                'status_code': status_code,
                **kwargs
            }
        )

class ConfigurationError(AIError):
    """خطای تنظیمات"""
    def __init__(self, message: str, config_key: str, **kwargs):
        super().__init__(
            message=message,
            error_code='CONFIG_ERROR',
            details={'config_key': config_key, **kwargs}
        )