"""
Training Module
--------------
این ماژول سیستم آموزش هوش مصنوعی را پیاده‌سازی می‌کند.
شامل بخش‌های مدیریت داده، ارزیابی و خط لوله آموزش است.
"""

from .pipeline import TrainingPipeline, TrainingConfig
from .evaluator import EvaluationResult, TrainingMetrics
from .data_loader import DataLoader, DataBatch, DatasetStats

__all__ = [
    'TrainingPipeline',
    'TrainingConfig',
    'EvaluationResult',
    'TrainingMetrics',
    'DataLoader',
    'DataBatch',
    'DatasetStats'
]