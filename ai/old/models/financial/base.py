"""
رابط‌ها و کلاس‌های پایه برای ماژول تحلیل مالی

این ماژول پایه‌ای‌ترین لایه از معماری تحلیل مالی را تعریف می‌کند. طبق اصول معماری تمیز،
این لایه شامل رابط‌ها و موجودیت‌های اصلی کسب و کار است و مستقل از جزئیات پیاده‌سازی است.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
from decimal import Decimal
from . import FinancialModelConfig, TimeFrame, AnalysisType


@dataclass
class MarketData:
    """موجودیت داده‌های بازار"""
    symbol: str
    timestamp: datetime
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal
    timeframe: TimeFrame
    metadata: Optional[Dict[str, Any]] = None

    def validate(self) -> bool:
        """اعتبارسنجی داده‌های بازار"""
        return all([
            self.open_price > 0,
            self.high_price >= self.open_price,
            self.low_price <= self.open_price,
            self.close_price > 0,
            self.volume >= 0
        ])


@dataclass
class TechnicalIndicator:
    """موجودیت شاخص تکنیکال"""
    name: str
    value: float
    timestamp: datetime
    settings: Dict[str, Any]
    confidence: float = 1.0


@dataclass
class AnalysisResult:
    """موجودیت نتیجه تحلیل"""
    market_data: MarketData
    indicators: List[TechnicalIndicator]
    analysis_type: AnalysisType
    signals: Dict[str, float]
    confidence: float
    analysis_time: float
    metadata: Optional[Dict[str, Any]] = None


class DataPreprocessor(ABC):
    """رابط پیش‌پردازش داده‌های بازار"""

    @abstractmethod
    async def preprocess(self, data: List[MarketData]) -> List[MarketData]:
        """پیش‌پردازش داده‌های خام"""
        pass

    @abstractmethod
    async def normalize(self, data: List[MarketData]) -> List[MarketData]:
        """نرمال‌سازی داده‌ها"""
        pass

    @abstractmethod
    async def validate_data(self, data: List[MarketData]) -> bool:
        """اعتبارسنجی داده‌ها"""
        pass


class TechnicalAnalyzer(ABC):
    """رابط تحلیل تکنیکال"""

    @abstractmethod
    async def calculate_indicators(self, data: List[MarketData]) -> List[TechnicalIndicator]:
        """محاسبه شاخص‌های تکنیکال"""
        pass

    @abstractmethod
    async def analyze_trend(self, data: List[MarketData]) -> Dict[str, float]:
        """تحلیل روند"""
        pass

    @abstractmethod
    async def detect_patterns(self, data: List[MarketData]) -> List[Dict[str, Any]]:
        """شناسایی الگوها"""
        pass


class SignalGenerator(ABC):
    """رابط تولید سیگنال"""

    @abstractmethod
    async def generate_signals(self, analysis: AnalysisResult) -> Dict[str, float]:
        """تولید سیگنال‌های معاملاتی"""
        pass

    @abstractmethod
    async def validate_signal(self, signal: Dict[str, float]) -> bool:
        """اعتبارسنجی سیگنال"""
        pass


class FinancialBaseModel(ABC):
    """کلاس پایه برای مدل‌های تحلیل مالی"""

    def __init__(self, config: FinancialModelConfig):
        self.config = config
        self._preprocessor: Optional[DataPreprocessor] = None
        self._analyzer: Optional[TechnicalAnalyzer] = None
        self._signal_generator: Optional[SignalGenerator] = None

    @abstractmethod
    async def initialize(self) -> None:
        """راه‌اندازی اولیه مدل"""
        pass

    @abstractmethod
    async def analyze(self, market_data: List[MarketData]) -> AnalysisResult:
        """تحلیل داده‌های بازار"""
        # اعتبارسنجی داده‌ها
        for data in market_data:
            if not data.validate():
                raise ValueError(f"Invalid market data for {data.symbol}")
        pass

    @abstractmethod
    async def backtest(self, historical_data: List[MarketData]) -> Dict[str, float]:
        """آزمایش مدل روی داده‌های تاریخی"""
        pass

    @abstractmethod
    async def optimize(self, training_data: List[MarketData]) -> None:
        """بهینه‌سازی پارامترهای مدل"""
        pass

    @abstractmethod
    async def save_state(self) -> bool:
        """ذخیره وضعیت مدل"""
        pass

    @abstractmethod
    async def load_state(self) -> bool:
        """بارگذاری وضعیت مدل"""
        pass


class FinancialModelFactory:
    """کارخانه ساخت مدل‌های تحلیل مالی"""

    @staticmethod
    async def create_model(config: FinancialModelConfig) -> FinancialBaseModel:
        """ایجاد نمونه جدید از مدل تحلیل مالی با توجه به تنظیمات"""
        # این متد در پیاده‌سازی‌های بعدی تکمیل می‌شود
        pass