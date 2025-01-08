# ai/models/nlp/__init__.py
"""
Natural Language Processing Models
--------------------------------
This package contains multilingual NLP models supporting:
- Persian (Farsi)
- English
- Arabic

The models handle tasks such as:
- Text classification
- Named Entity Recognition
- Sentiment Analysis
- Language Detection and Translation
"""
"""
ماژول پردازش زبان طبیعی (NLP Module)

این ماژول مسئول پردازش متن و تحلیل زبان طبیعی در زبان‌های مختلف است.
قابلیت‌های اصلی شامل پردازش چندزبانه، تحلیل احساسات، و یادگیری از مدل‌های بزرگتر است.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum
import torch
import logging

logger = logging.getLogger(__name__)

class SupportedLanguage(Enum):
    """زبان‌های پشتیبانی شده"""
    PERSIAN = 'fa'
    ENGLISH = 'en'
    ARABIC = 'ar'

@dataclass
class NLPModelConfig:
    """تنظیمات پایه برای مدل‌های NLP"""
    name: str
    languages: List[SupportedLanguage]
    max_sequence_length: int = 512
    batch_size: int = 32
    num_workers: int = 4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path: str = None
    tokenizer_path: str = None

# تنظیمات پیش‌فرض برای مدل NLP
DEFAULT_CONFIG = NLPModelConfig(
    name="multilingual-base",
    languages=[SupportedLanguage.PERSIAN, SupportedLanguage.ENGLISH, SupportedLanguage.ARABIC],
)

# تنظیمات مربوط به توکنایزر
TOKENIZER_CONFIG = {
    'add_special_tokens': True,
    'padding': True,
    'truncation': True,
    'return_attention_mask': True
}

# تنظیمات مربوط به اسکیل‌پذیری مدل NLP
SCALING_CONFIG = {
    'metrics': {
        'cpu_threshold': 80,  # درصد
        'memory_threshold': 85,  # درصد
        'request_queue_size': 1000,
        'latency_threshold': 300  # میلی‌ثانیه
    },
    'resources': {
        'min_cpu': '1',
        'max_cpu': '4',
        'min_memory': '4Gi',
        'max_memory': '16Gi',
        'gpu_required': torch.cuda.is_available()
    }
}

# استراتژی‌های یادگیری و بهبود مدل
LEARNING_CONFIG = {
    'learning_rate': 1e-5,
    'warmup_steps': 1000,
    'max_epochs': 10,
    'evaluation_strategy': 'steps',
    'eval_steps': 500,
    'save_strategy': 'epoch',
    'logging_steps': 100
}

logger.info(f"NLP Module initialized with default config: {DEFAULT_CONFIG}")
logger.info(f"Using device: {DEFAULT_CONFIG.device}")
logger.info(f"Supported languages: {[lang.value for lang in SupportedLanguage]}")