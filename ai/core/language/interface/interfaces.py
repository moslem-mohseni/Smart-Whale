"""
Language Processing Interfaces
--------------------------
تعریف اینترفیس‌های پایه برای پردازش زبان
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ProcessingResult:
    """نتیجه پردازش متن"""
    text: str
    tokens: List[str]
    confidence: float
    language: str
    features: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime = datetime.now()

    def is_confident(self) -> bool:
        """بررسی کافی بودن اطمینان"""
        return self.confidence >= 0.7