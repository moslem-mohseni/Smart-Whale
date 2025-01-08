"""
Base Language Processing Module
-----------------------------
این ماژول کلاس‌های پایه برای پردازش و یادگیری زبان را تعریف می‌کند.
تمام پردازشگرها و یادگیرنده‌های زبان‌های مختلف باید از این کلاس‌ها ارث‌بری کنند.
"""

from .language_processor import LanguageProcessor, ProcessingResult
from .language_learner import LanguageLearner, LearningExample

__all__ = [
    'LanguageProcessor',
    'ProcessingResult',
    'LanguageLearner',
    'LearningExample'
]