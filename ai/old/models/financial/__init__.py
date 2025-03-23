# ai/models/financial/__init__.py
"""
Financial Analysis Models
-----------------------
Models for financial data analysis and prediction including:
- Technical Analysis
- Price Prediction
- Market Sentiment Analysis
- Risk Assessment

These models work in conjunction with the NLP models to provide
comprehensive market analysis based on both technical and textual data.
"""

"""
ماژول تحلیل مالی (Financial Analysis Module)

این ماژول مسئول تحلیل تکنیکال بازارهای مالی است. این مدل از ترکیبی از
تحلیل‌های کلاسیک و یادگیری ماشین برای شناسایی الگوها و پیش‌بینی روندها استفاده می‌کند.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum
import torch
import logging

logger = logging.getLogger(__name__)

class TimeFrame(Enum):
    """بازه‌های زمانی تحلیل"""
    MINUTE_1 = '1m'
    MINUTE_5 = '5m'
    MINUTE_15 = '15m'
    MINUTE_30 = '30m'
    HOUR_1 = '1h'
    HOUR_4 = '4h'
    DAY_1 = '1d'
    WEEK_1 = '1w'

class AnalysisType(Enum):
    """انواع تحلیل‌های قابل انجام"""
    TREND = 'trend'
    PATTERN = 'pattern'
    MOMENTUM = 'momentum'
    VOLATILITY = 'volatility'
    VOLUME = 'volume'
    SUPPORT_RESISTANCE = 'support_resistance'

@dataclass
class FinancialModelConfig:
    """تنظیمات پایه برای مدل‌های تحلیل مالی"""
    name: str
    timeframes: List[TimeFrame]
    analysis_types: List[AnalysisType]
    lookback_periods: int = 100
    batch_size: int = 64
    num_workers: int = 4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path: str = None

# تنظیمات پیش‌فرض برای مدل تحلیل مالی
DEFAULT_CONFIG = FinancialModelConfig(
    name="technical-analysis-base",
    timeframes=[
        TimeFrame.MINUTE_15,
        TimeFrame.HOUR_1,
        TimeFrame.DAY_1
    ],
    analysis_types=[
        AnalysisType.TREND,
        AnalysisType.PATTERN,
        AnalysisType.MOMENTUM
    ]
)

# تنظیمات مربوط به اندیکاتورهای تکنیکال
INDICATOR_CONFIG = {
    'moving_averages': ['SMA', 'EMA', 'WMA'],
    'oscillators': ['RSI', 'MACD', 'Stochastic'],
    'volatility': ['ATR', 'Bollinger'],
    'volume': ['OBV', 'Volume Profile'],
    'trend': ['ADX', 'Supertrend']
}

# تنظیمات اسکیل‌پذیری مدل مالی
SCALING_CONFIG = {
    'metrics': {
        'cpu_threshold': 75,  # درصد
        'memory_threshold': 80,  # درصد
        'request_queue_size': 500,
        'latency_threshold': 200  # میلی‌ثانیه
    },
    'resources': {
        'min_cpu': '1',
        'max_cpu': '4',
        'min_memory': '2Gi',
        'max_memory': '8Gi',
        'gpu_required': False  # تحلیل تکنیکال معمولاً نیاز به GPU ندارد
    }
}

# تنظیمات مربوط به سیگنال‌دهی
SIGNAL_CONFIG = {
    'min_confidence': 0.75,
    'signal_interval': 60,  # ثانیه
    'max_signals_per_interval': 10,
    'risk_levels': ['low', 'medium', 'high']
}

logger.info(f"Financial Analysis Module initialized with default config: {DEFAULT_CONFIG}")
logger.info(f"Using device: {DEFAULT_CONFIG.device}")
logger.info(f"Active timeframes: {[tf.value for tf in DEFAULT_CONFIG.timeframes]}")
logger.info(f"Active analysis types: {[at.value for at in DEFAULT_CONFIG.analysis_types]}")