from ai.models.language.context.analyzer.vector_search import VectorSearch

class RetrieverVectorSearch:
    """
    این کلاس یک wrapper برای `VectorSearch` در `analyzer` است که برای بازیابی اطلاعات در `retriever` استفاده می‌شود.
    """

    def __init__(self, similarity_threshold: float = 0.75):
        """
        مقداردهی اولیه `RetrieverVectorSearch`
        :param similarity_threshold: حد آستانه‌ی شباهت برای بازیابی اطلاعات (بین `0` و `1`، پیش‌فرض `0.75`)
        """
        self.vector_search = VectorSearch(similarity_threshold)

    async def find_related_messages(self, query: str, top_n: int = 5):
        """
        بازیابی پیام‌های مرتبط از مکالمات ذخیره‌شده بر اساس جستجوی برداری.
        """
        return await self.vector_search.find_similar(query, top_n)
