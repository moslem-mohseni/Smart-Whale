import pytest
import torch
from ...ai.core.language.external.parsbert_adapter.adapter import ParsBERTAdapter
from ...ai.core.language.external.parsbert_adapter.processor import ParsBERTProcessor


@pytest.mark.asyncio
async def test_initialize():
    adapter = ParsBERTAdapter()
    success = await adapter.initialize()
    assert success, "Model initialization failed"
    assert adapter.model is not None, "Model should be loaded"
    assert adapter.tokenizer is not None, "Tokenizer should be loaded"


@pytest.mark.asyncio
async def test_process_text():
    adapter = ParsBERTAdapter()
    await adapter.initialize()
    result = await adapter.process_text("این یک تست است")
    assert isinstance(result, torch.Tensor), "Output should be a tensor"
    assert result.shape[0] == 1, "Output should have batch dimension"


@pytest.mark.asyncio
async def test_batch_process():
    adapter = ParsBERTAdapter()
    await adapter.initialize()
    processor = ParsBERTProcessor(adapter)
    texts = ["متن اول", "متن دوم", "متن سوم"]
    results = await processor.batch_process(texts)
    assert len(results) == len(texts), "Batch output size should match input size"
    assert all(isinstance(res, torch.Tensor) for res in results if res is not None), "Each output should be a tensor"


@pytest.mark.asyncio
async def test_prefetch_texts():
    adapter = ParsBERTAdapter()
    await adapter.initialize()
    texts = ["نمونه متن ۱", "نمونه متن ۲"]
    await adapter.prefetch_texts(texts)
    assert True, "Prefetch should execute without errors"


@pytest.mark.asyncio
async def test_memory_management():
    adapter = ParsBERTAdapter()
    await adapter.initialize()
    try:
        for _ in range(100):
            await adapter.process_text("این یک تست طولانی برای بررسی مدیریت حافظه است")
    except torch.cuda.OutOfMemoryError:
        pytest.fail("Model should manage memory without running out of GPU memory")
