# persian/language_processors/literature/literature_vector.py
"""
ماژول literature_vector.py

این ماژول شامل کلاس LiteratureVectorManager برای ایجاد بردار معنایی متون ادبی و جستجوی بردارهای مشابه می‌باشد.
این کلاس از مدل‌های هوشمند (یا مدل معلم) برای تولید بردار استفاده می‌کند و در صورت عدم موفقیت، به تولید بردار تصادفی می‌پردازد.
همچنین از MilvusAdapter برای درج و جستجوی بردارها به صورت asynchronous استفاده می‌کند.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
import asyncio

try:
    import torch
except ImportError:
    torch = None

# استفاده از MilvusAdapter به صورت بدون تغییر
from ai.models.language.infrastructure.vector_store.milvus_adapter import MilvusAdapter
from ai.models.language.infrastructure.vector_store.vector_search import VectorSearch
from ai.models.language.infrastructure.vector_store.milvus_adapter import VectorService

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LiteratureVectorManager:
    def __init__(self,
                 smart_model: Optional[Any] = None,
                 teacher_model: Optional[Any] = None,
                 vector_store: Optional[MilvusAdapter] = None,
                 vector_dim: int = 128):
        """
        سازنده LiteratureVectorManager.

        Args:
            smart_model (Optional[Any]): مدل هوشمند برای تولید بردار معنایی متون ادبی.
            teacher_model (Optional[Any]): مدل معلم برای تولید بردار معنایی متون ادبی.
            vector_store (Optional[MilvusAdapter]): شیء مدیریت بردار در vector store.
            vector_dim (int): ابعاد بردار معنایی (پیش‌فرض 128).
        """
        self.smart_model = smart_model
        self.teacher_model = teacher_model
        # اگر vector_store داده نشده باشد، از یک نمونه پیش‌فرض MilvusAdapter استفاده می‌کنیم.
        # فرض می‌کنیم VectorSearch قابلیت استفاده به عنوان vector_service را دارد.
        dummy_vector_service = VectorService()

        self.vector_store = vector_store if vector_store is not None else MilvusAdapter(
            vector_service=dummy_vector_service)
        self.vector_dim = vector_dim

    def create_semantic_vector(self, text: str) -> List[float]:
        """
        ایجاد بردار معنایی برای یک متن ادبی با استفاده از مدل‌های هوشمند یا fallback به بردار تصادفی.

        Args:
            text (str): متن ادبی.

        Returns:
            List[float]: بردار معنایی به صورت لیستی از اعداد.
        """
        vector = None

        # تلاش برای استفاده از smart_model
        if self.smart_model and hasattr(self.smart_model, "confidence_level"):
            try:
                confidence = self.smart_model.confidence_level(text)
                if confidence >= 0.5:
                    if torch is not None:
                        with torch.no_grad():
                            model_vector = self.smart_model.forward(text)
                            if hasattr(model_vector, "cpu"):
                                vector = model_vector.cpu().numpy().tolist()
                            else:
                                vector = model_vector
                    else:
                        vector = self.smart_model.forward(text)
            except Exception as e:
                logger.error(f"Error creating vector with smart model: {e}")

        # اگر smart_model ناموفق بود، استفاده از teacher_model
        if vector is None and self.teacher_model and hasattr(self.teacher_model, "forward"):
            try:
                if torch is not None:
                    with torch.no_grad():
                        teacher_vector = self.teacher_model.forward(text)
                        if hasattr(teacher_vector, "cpu"):
                            vector = teacher_vector.cpu().numpy().tolist()
                        else:
                            vector = teacher_vector
                else:
                    vector = self.teacher_model.forward(text)
            except Exception as e:
                logger.error(f"Error creating vector with teacher model: {e}")

        # fallback: تولید بردار تصادفی
        if vector is None:
            logger.warning("Generating random semantic vector as fallback.")
            vector = np.random.randn(self.vector_dim).tolist()

        return vector

    async def insert_vector(self, vector_data: Dict[str, Any]) -> bool:
        """
        درج بردار معنایی متن ادبی در vector store.

        Args:
            vector_data (Dict[str, Any]): دیکشنری شامل "id" و "vector".

        Returns:
            bool: True در صورت موفقیت، False در غیر این صورت.
        """
        try:
            # توجه: MilvusAdapter متد insert_vectors را به صورت async فراخوانی می‌کند.
            await self.vector_store.insert_vectors("literary_vectors", [vector_data["vector"]])
            logger.info(f"Vector inserted for id {vector_data.get('id')}.")
            return True
        except Exception as e:
            logger.error(f"Error inserting vector: {e}")
            return False

    async def search_vectors(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        جستجوی بردارهای معنایی مشابه برای یک متن ادبی در vector store.

        Args:
            query_vector (List[float]): بردار معنایی برای جستجو.
            top_k (int): تعداد نتایج برتر.

        Returns:
            List[Dict[str, Any]]: لیستی از نتایج جستجو شامل id، distance و سایر اطلاعات.
        """
        try:
            results = await self.vector_store.search_vectors("literary_vectors", query_vector, top_k)
            logger.info(f"Found {len(results)} vectors matching the query.")
            return results if results is not None else []
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []


# تست نمونه
if __name__ == "__main__":
    import asyncio


    async def main():
        sample_text = "در این متن ادبی، فضای معنایی و زیبایی خاصی به چشم می‌خورد."
        vector_manager = LiteratureVectorManager()
        semantic_vector = vector_manager.create_semantic_vector(sample_text)
        print("Semantic vector:", semantic_vector)

        vector_data = {
            "id": "lit_sample_001",
            "vector": semantic_vector
        }
        inserted = await vector_manager.insert_vector(vector_data)
        print("Insert vector result:", inserted)

        search_results = await vector_manager.search_vectors(semantic_vector, top_k=3)
        print("Search results:", search_results)


    asyncio.run(main())
