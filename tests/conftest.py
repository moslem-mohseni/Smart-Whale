# tests/conftest.py
import pytest
from pathlib import Path
from typing import Dict, Any

class LanguageTestHelper:
    """کلاس کمکی برای تست‌های پردازش زبان"""
    
    def get_test_config(self, language: str) -> Dict[str, Any]:
        """دریافت تنظیمات تست برای زبان مورد نظر"""
        base_config = {
            'model_management': {
                'max_models': 2,
                'default_batch_size': 16,
                'timeout': 10,
            },
            'knowledge': {
                'max_cache_size': 100,
                'cache_ttl': 300,
            },
            'learning': {
                'min_confidence': 0.6,
                'max_retries': 2,
            }
        }
        
        if language == 'fa':
            base_config.update({
                'hazm_resources_path': str(Path(__file__).parent / 'resources' / 'hazm'),
                'parsbert_model': 'HooshvareLab/bert-fa-base-uncased'
            })
            
        return base_config

    def verify_text_processing_result(self, original_text: str, result: Any) -> None:
        """بررسی صحت نتایج پردازش متن"""
        assert result is not None, "نتیجه پردازش نباید None باشد"
        assert result.text == original_text, "متن اصلی باید حفظ شود"
        assert result.confidence >= 0, "اطمینان باید مثبت باشد"
        assert result.confidence <= 1, "اطمینان نباید از 1 بیشتر باشد"
        assert len(result.tokens) > 0, "باید حداقل یک توکن وجود داشته باشد"
        assert result.features is not None, "ویژگی‌ها نباید None باشند"

@pytest.fixture
def language_test_helper():
    """فیکسچر برای دسترسی به توابع کمکی تست زبان"""
    return LanguageTestHelper()

@pytest.fixture
def test_data_path():
    """فیکسچر برای دسترسی به مسیر داده‌های تست"""
    return Path(__file__).parent / 'test_data'