# persian/language_processors/proverbs/proverb_vector.py
"""
ماژول proverb_vector.py

این ماژول شامل توابع و کلاس‌هایی برای ایجاد بردار معنایی ضرب‌المثل و جستجوی بردارهای معنایی می‌باشد.
امکانات اصلی:
  - ایجاد بردار معنایی ضرب‌المثل با استفاده از مدل‌های هوشمند یا fallback.
  - یکپارچه‌سازی با زیرساخت‌های vector store (مثلاً Milvus) جهت درج و جستجوی بردارها.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
import asyncio

try:
    import torch
except ImportError:
    torch = None

# فرض بر این است که MilvusAdapter تغییرناپذیر است و از آن استفاده می‌کنیم
from ai.models.language.infrastructure.vector_store.milvus_adapter import MilvusAdapter
from ai.models.language.infrastructure.vector_store.vector_search import VectorSearch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ProverbVectorManager:
    def __init__(self,
                 smart_model: Optional[Any] = None,
                 teacher_model: Optional[Any] = None,
                 vector_store: Optional[MilvusAdapter] = None,
                 vector_dim: int = 128):
        """
        سازنده ProverbVectorManager.

        Args:
            smart_model (Optional[Any]): مدل هوشمند برای تولید بردار معنایی ضرب‌المثل.
            teacher_model (Optional[Any]): مدل معلم برای تولید بردار معنایی ضرب‌المثل.
            vector_store (Optional[MilvusAdapter]): شیء مدیریت بردار در vector store.
            vector_dim (int): ابعاد بردار معنایی (پیش‌فرض 128).
        """
        self.smart_model = smart_model
        self.teacher_model = teacher_model
        # اگر vector_store داده نشده باشد، یک نمونه از MilvusAdapter ایجاد می‌شود
        # توجه: vector_store در MilvusAdapter به صورت async عمل می‌کند.
        self.vector_store = vector_store
        self.vector_dim = vector_dim

    def create_semantic_vector(self, proverb: str, meaning: str) -> List[float]:
        """
        ایجاد بردار معنایی ضرب‌المثل با استفاده از مدل‌های هوشمند یا fallback.

        Args:
            proverb (str): متن ضرب‌المثل.
            meaning (str): معنی ضرب‌المثل.

        Returns:
            List[float]: بردار معنایی به عنوان لیستی از اعداد.
        """
        vector = None

        # تلاش برای استفاده از مدل هوشمند (smart model)
        if self.smart_model and hasattr(self.smart_model, "confidence_level"):
            try:
                confidence = self.smart_model.confidence_level(proverb)
                if confidence >= 0.5:
                    if torch is not None:
                        with torch.no_grad():
                            model_vector = self.smart_model.forward(f"{proverb} {meaning}")
                            if hasattr(model_vector, "cpu"):
                                vector = model_vector.cpu().numpy().tolist()
                            else:
                                vector = model_vector
                    else:
                        vector = self.smart_model.forward(f"{proverb} {meaning}")
            except Exception as e:
                logger.error(f"Error creating vector with smart model: {e}")

        # اگر smart_model ناموفق بود، استفاده از teacher_model
        if vector is None and self.teacher_model and hasattr(self.teacher_model, "forward"):
            try:
                if torch is not None:
                    with torch.no_grad():
                        teacher_vector = self.teacher_model.forward(f"{proverb} {meaning}")
                        if hasattr(teacher_vector, "cpu"):
                            vector = teacher_vector.cpu().numpy().tolist()
                        else:
                            vector = teacher_vector
                else:
                    vector = self.teacher_model.forward(f"{proverb} {meaning}")
            except Exception as e:
                logger.error(f"Error creating vector with teacher model: {e}")

        # در صورت عدم موفقیت، تولید بردار تصادفی
        if vector is None:
            logger.warning("Generating random semantic vector as fallback.")
            vector = np.random.randn(self.vector_dim).tolist()

        return vector

    async def insert_vector(self, vector_data: Dict[str, Any]) -> bool:
        """
        درج بردار معنایی ضرب‌المثل در vector store.

        Args:
            vector_data (Dict[str, Any]): اطلاعات بردار شامل "id" و "vector".

        Returns:
            bool: True در صورت موفقیت، False در غیر این صورت.
        """
        try:
            # فراخوانی async insert_vectors؛ توجه کنید که تنها بردار را ارسال می‌کنیم.
            await self.vector_store.insert_vectors("proverb_vectors", [vector_data["vector"]])
            logger.info(f"Vector inserted for id {vector_data.get('id')}.")
            return True
        except Exception as e:
            logger.error(f"Error inserting vector: {e}")
            return False

    async def search_vectors(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        جستجوی بردارهای معنایی مشابه در vector store.

        Args:
            query_vector (List[float]): بردار معنایی برای جستجو.
            top_k (int): تعداد نتایج برتر.

        Returns:
            List[Dict[str, Any]]: لیستی از نتایج جستجو شامل id، distance و سایر اطلاعات.
        """
        try:
            results = await self.vector_store.search_vectors("proverb_vectors", query_vector, top_k)
            logger.info(f"Found {len(results)} vectors matching the query.")
            return results if results is not None else []
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []


# تست نمونه
if __name__ == "__main__":
    import asyncio

    async def main():
        sample_proverb = "هر که بامش بیش برفش بیشتر"
        sample_meaning = "هر کس با دارایی بیشتر، مسئولیت و مشکلات بیشتری دارد"
        vector_manager = ProverbVectorManager()
        semantic_vector = vector_manager.create_semantic_vector(sample_proverb, sample_meaning)
        print("Semantic vector:", semantic_vector)

        # تست درج بردار (این متد async است)
        vector_data = {
            "id": "p_sample",
            "vector": semantic_vector
        }
        inserted = await vector_manager.insert_vector(vector_data)
        print("Insert vector result:", inserted)

        # تست جستجوی برداری
        search_results = await vector_manager.search_vectors(semantic_vector, top_k=3)
        print("Search results:", search_results)

    asyncio.run(main())
