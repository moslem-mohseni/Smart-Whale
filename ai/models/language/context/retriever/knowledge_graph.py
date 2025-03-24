import logging
import networkx as nx
from typing import Dict, List, Optional
from ai.models.language.context.retriever.fact_checker import FactChecker

class KnowledgeGraph:
    """
    این کلاس داده‌های معتبر را در قالب یک نمودار دانش سازمان‌دهی کرده و برای بازیابی اطلاعات از آن استفاده می‌کند.
    """

    def __init__(self):
        self.graph = nx.DiGraph()  # استفاده از یک گراف جهت‌دار
        self.fact_checker = FactChecker()
        logging.info("✅ KnowledgeGraph مقداردهی شد.")

    async def add_relationship(self, entity1: str, relation: str, entity2: str):
        """
        افزودن یک رابطه به نمودار دانش.

        :param entity1: اولین موجودیت (نود گراف)
        :param relation: نوع رابطه‌ی بین دو موجودیت
        :param entity2: دومین موجودیت (نود گراف)
        """
        self.graph.add_edge(entity1, entity2, relation=relation)
        logging.info(f"🔗 رابطه اضافه شد: {entity1} → {relation} → {entity2}")

    async def build_graph_from_context(self, user_id: str, chat_id: str, query: str):
        """
        دریافت داده‌های معتبر و تبدیل آن‌ها به یک نمودار دانش.

        :param user_id: شناسه‌ی کاربر
        :param chat_id: شناسه‌ی چت
        :param query: پیام مورد جستجو
        """
        context_data = await self.fact_checker.validate_context(user_id, chat_id, query)
        if not context_data:
            logging.warning(f"⚠️ داده‌ی معتبری برای کاربر {user_id} و چت {chat_id} یافت نشد.")
            return

        # تبدیل داده‌های معتبر به روابط گراف
        for key, value in context_data["validated_data"].items():
            await self.add_relationship(user_id, key, value)

    async def find_related_entities(self, entity: str, depth: int = 2) -> Optional[List[str]]:
        """
        یافتن موجودیت‌های مرتبط با یک داده‌ی خاص.

        :param entity: نام موجودیت
        :param depth: عمق جستجو در نمودار دانش
        :return: لیستی از موجودیت‌های مرتبط (در صورت وجود)
        """
        if entity not in self.graph:
            logging.warning(f"⚠️ موجودیت '{entity}' در نمودار دانش یافت نشد.")
            return None

        related_entities = list(nx.single_source_shortest_path(self.graph, entity, cutoff=depth).keys())
        logging.info(f"🔍 موجودیت‌های مرتبط با '{entity}': {related_entities}")
        return related_entities

    async def get_relationships(self, entity: str) -> Optional[List[Dict]]:
        """
        دریافت تمامی روابط مرتبط با یک موجودیت خاص.

        :param entity: نام موجودیت
        :return: لیستی از روابط موجودیت (در صورت وجود)
        """
        if entity not in self.graph:
            logging.warning(f"⚠️ موجودیت '{entity}' در نمودار دانش یافت نشد.")
            return None

        relationships = [
            {"source": entity, "relation": data["relation"], "target": target}
            for target, data in self.graph[entity].items()
        ]
        logging.info(f"📊 روابط مرتبط با '{entity}': {relationships}")
        return relationships
