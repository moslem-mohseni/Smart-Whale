"""
tests/unit/language/test_persian.py
----------------------------------
تست‌های واحد برای سیستم پردازش زبان فارسی

مسیردهی‌های اصلی:
- resources/hazm/: مدل‌های hazm
- resources/knowledge/: پایگاه دانش
- resources/models/: مدل‌های پردازش زبان
"""

from typing import Dict, Any
import pytest
import logging
import json
import shutil
from pathlib import Path
from datetime import datetime

from ai.core.language import get_language_processor, get_language_learner
from ai.core.language.persian.knowledge_base import PersianKnowledgeBase

logger = logging.getLogger('language_tests')


@pytest.fixture(autouse=True)
async def setup_test_resources():  # حذف scope="session"
    resources_path = Path(__file__).parent / 'resources'

    hazm_path = resources_path / 'hazm'
    hazm_path.mkdir(parents=True, exist_ok=True)
    for model in ['postagger.model', 'chunker.model']:
        with open(hazm_path / model, 'wb') as f:
            f.write(b'dummy model data')

    kb_path = resources_path / 'knowledge'
    kb_path.mkdir(parents=True, exist_ok=True)
    with open(kb_path / 'persian_knowledge.json', 'w', encoding='utf-8') as f:
        json.dump({
            'concepts': {},
            'patterns': {},
            'stats': {
                'total_concepts': 0,
                'total_patterns': 0,
                'total_updates': 0,
                'last_update': None
            }
        }, f, ensure_ascii=False, indent=2)

    yield
    shutil.rmtree(resources_path, ignore_errors=True)


@pytest.mark.language
class TestPersianLanguageProcessing:
    @pytest.fixture(autouse=True)
    async def setup(self, language_test_helper, setup_test_resources):
        self.config = language_test_helper.get_test_config('fa')
        self.config.update({
            'hazm_resources_path': Path(__file__).parent / 'resources' / 'hazm',
            'models_path': Path(__file__).parent / 'resources' / 'models',
            'storage_path': Path(__file__).parent / 'resources' / 'knowledge'
        })
        self.helper = language_test_helper
        self.processor = get_language_processor('fa', self.config)
        self.learner = get_language_learner('fa', self.config)
        self.knowledge_base = PersianKnowledgeBase(self.config)

        await self.processor.initialize()
        await self.learner.initialize()
        await self.knowledge_base.load_knowledge()

        self.test_texts = {
            'simple': 'این یک جمله ساده برای تست است.',
            'complex': 'هوش مصنوعی در حال تغییر دنیای فناوری است و ما باید خود را با آن تطبیق دهیم.',
            'technical': 'پردازش زبان طبیعی یکی از شاخه‌های مهم یادگیری ماشین است.',
            'poetic': 'برگ‌های پاییزی در باد می‌رقصند و داستان فصل را روایت می‌کنند.'
        }

        yield
        await self.cleanup()

    async def cleanup(self):
        try:
            await self.knowledge_base.save_knowledge()
        except Exception as e:
            logger.warning(f"Error in cleanup: {e}")

    @pytest.mark.asyncio
    @pytest.mark.language
    async def test_basic_text_processing(self):
        result = await self.processor.process(self.test_texts['simple'])

        assert result is not None
        assert result.language == 'fa'
        assert result.text == self.test_texts['simple']
        assert len(result.tokens) >= 7
        assert result.confidence >= 0.5

    @pytest.mark.asyncio
    @pytest.mark.language
    async def test_deep_analysis(self):
        result = await self.processor.analyze_deeply(self.test_texts['complex'])

        assert result is not None
        assert 'semantic' in result.features
        assert 'structural' in result.features
        assert result.features['semantic']['embeddings'] is not None
        assert result.features['semantic']['embeddings'].size > 0

    @pytest.mark.asyncio
    @pytest.mark.language
    async def test_learning_capability(self):
        initial = await self.processor.process(self.test_texts['technical'])
        await self.learner.learn(
            text=self.test_texts['technical'],
            features=initial.features,
            source='test'
        )
        final = await self.processor.process(self.test_texts['technical'])

        assert final.confidence > initial.confidence
        assert len(final.features) >= len(initial.features)

    @pytest.mark.asyncio
    @pytest.mark.language
    async def test_knowledge_base_management(self):
        pattern = "[فعل] می‌کنند"
        await self.knowledge_base.add_pattern(
            pattern=pattern,
            example=self.test_texts['poetic'],
            confidence=0.9
        )
        matches = await self.knowledge_base.find_matching_patterns(
            "پرندگان در آسمان پرواز می‌کنند",
            min_confidence=0.5
        )

        assert matches, "حداقل یک الگو باید پیدا شود"
        found = matches[0]
        assert found.pattern == pattern
        assert found.confidence >= 0.9

    @pytest.mark.asyncio
    @pytest.mark.language
    async def test_error_handling(self):
        with pytest.raises(ValueError):
            await self.processor.process("")

        english_result = await self.processor.process("This is English text")
        assert english_result.confidence < 0.3

        mixed_result = await self.processor.process("این یک mixed متن است")
        assert 0.3 <= mixed_result.confidence <= 0.7


