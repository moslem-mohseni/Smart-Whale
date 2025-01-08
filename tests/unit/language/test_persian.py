"""Persian Language Processing Tests"""
from typing import Dict, Any
import pytest
import logging
from pathlib import Path

from ai.core.language import get_language_processor, get_language_learner
from ai.core.language.persian.knowledge_base import PersianKnowledgeBase

logger = logging.getLogger('language_tests')


@pytest.mark.language
class TestPersianLanguageProcessing:
    @pytest.fixture(autouse=True)
    async def setup(self, language_test_helper):
        self.config = language_test_helper.get_test_config('fa')
        self.config.update({
            'hazm_resources_path': Path(__file__).parent / 'resources' / 'hazm',
            'models_path': Path(__file__).parent / 'resources' / 'models'
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
        self.helper.verify_text_processing_result(self.test_texts['simple'], result)
        assert result.language == 'fa'
        assert len(result.tokens) >= 7

    @pytest.mark.asyncio
    @pytest.mark.language
    async def test_deep_analysis(self):
        result = await self.processor.analyze_deeply(self.test_texts['complex'])
        assert 'semantic' in result.features
        assert 'structural' in result.features
        semantic = result.features['semantic']
        assert semantic['embeddings'] is not None
        assert semantic['embeddings'].size > 0

    @pytest.mark.asyncio
    @pytest.mark.language
    async def test_learning_capability(self):
        initial = await self.processor.process(self.test_texts['technical'])
        await self.learner.learn(self.test_texts['technical'], initial.features, 'test')
        final = await self.processor.process(self.test_texts['technical'])

        assert final.confidence > initial.confidence
        assert len(final.features) >= len(initial.features)

    @pytest.mark.asyncio
    @pytest.mark.language
    async def test_knowledge_base_management(self):
        pattern = "[فعل] می‌کنند"
        await self.knowledge_base.add_pattern(pattern, self.test_texts['poetic'], 0.9)

        matches = await self.knowledge_base.find_matching_patterns("پرندگان در آسمان پرواز می‌کنند")
        assert matches
        found = matches[0]
        assert found.pattern == pattern
        assert found.confidence >= 0.9
        assert found.confidence >= 0.9

    @pytest.mark.asyncio
    @pytest.mark.language
    async def test_error_handling(self):
        with pytest.raises(ValueError):
            await self.processor.process("")

        english = await self.processor.process("This is English text")
        assert english.confidence < 0.3

        mixed = await self.processor.process("این یک mixed متن است")
        assert 0.3 <= mixed.confidence <= 0.7


