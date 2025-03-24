# persian/language_processors/domain/domain_data.py

"""
ماژول domain_data.py

این ماژول شامل کلاس DomainDataAccess است که وظیفه دسترسی به داده‌های حوزه‌های تخصصی (domains)،
مفاهیم (concepts)، روابط (relations) و ویژگی‌ها (attributes) را بر عهده دارد. داده‌ها از
پایگاه داده ClickHouse (با استفاده از ClickHouseAdapter) بارگذاری یا ذخیره می‌شوند و
برای کش‌گذاری از Redis (با استفاده از RedisAdapter) بهره می‌بریم.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional

# فرض بر این است که فایل‌های زیر در مسیرهای مربوطه قرار دارند
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseAdapter
from ai.models.language.infrastructure.caching.redis_adapter import RedisAdapter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DomainDataAccess:
    """
    کلاس DomainDataAccess

    این کلاس عملیات اصلی CRUD روی جداول domain, domain_concepts, concept_relations, concept_attributes
    را انجام می‌دهد و همچنین داده‌ها را در Redis کش می‌کند.
    """

    def __init__(self):
        """
        سازنده DomainDataAccess
        در اینجا اتصال به ClickHouseAdapter و RedisAdapter برقرار می‌شود.
        """
        try:
            # اتصال به پایگاه داده (ClickHouse) و Redis
            self.database = ClickHouseAdapter()
            self.redis = RedisAdapter()
            logger.info("DomainDataAccess: اتصال به پایگاه داده و Redis برقرار شد.")
        except Exception as e:
            logger.error(f"DomainDataAccess: خطا در اتصال به پایگاه داده یا Redis: {e}")
            raise

    async def connect_resources(self) -> None:
        """
        در صورتی که نیاز به متد اتصال جداگانه باشد، می‌توان این متد را فراخوانی کرد.
        """
        try:
            # اگر نیاز به متد connect در ClickHouseAdapter داشتید اینجا فراخوانی کنید
            # await self.database.connect()
            await self.redis.connect()
            logger.info("DomainDataAccess: منابع ClickHouse و Redis با موفقیت متصل شدند.")
        except Exception as e:
            logger.error(f"خطا در اتصال منابع: {e}")
            raise

    async def disconnect_resources(self) -> None:
        """
        قطع اتصال از منابع پایگاه داده و Redis
        """
        try:
            # await self.database.disconnect()
            await self.redis.disconnect()
            logger.info("DomainDataAccess: منابع با موفقیت قطع اتصال شدند.")
        except Exception as e:
            logger.error(f"خطا در قطع اتصال منابع: {e}")

    async def array_to_sql(self, array: List[str]) -> str:
        """
        تبدیل آرایه به فرمت مناسب SQL برای ClickHouse (برای ستون‌های Array(String))
        """
        if not array:
            return "[]"
        escaped_items = []
        for item in array:
            # جایگزینی ' در رشته‌ها
            escaped = item.replace("'", "''")
            escaped_items.append(f"'{escaped}'")
        items_str = ", ".join(escaped_items)
        return f"[{items_str}]"

    # ------------------------------------------------------------------------
    # --------------------------- DOMAINS -------------------------------------
    # ------------------------------------------------------------------------

    async def load_domains(self) -> Dict[str, Dict[str, Any]]:
        """
        بارگذاری تمامی حوزه‌های تخصصی از پایگاه داده.
        """
        query = """
            SELECT
                domain_id,
                domain_name,
                domain_code,
                parent_domain,
                description,
                popularity,
                source,
                discovery_time
            FROM domains
        """
        try:
            rows = await self.database.execute(query)
            if rows:
                domains = {row['domain_id']: row for row in rows}
                logger.info(f"DomainDataAccess: {len(domains)} حوزه بازیابی شدند.")
                return domains
            else:
                logger.info("DomainDataAccess: هیچ حوزه‌ای یافت نشد.")
                return {}
        except Exception as e:
            logger.error(f"DomainDataAccess: خطا در بارگذاری حوزه‌ها: {e}")
            return {}

    async def store_domain(self, domain: Dict[str, Any]) -> bool:
        """
        ذخیره یا به‌روزرسانی یک حوزه در پایگاه داده.
        """
        check_query = f"""
            SELECT domain_id FROM domains
            WHERE domain_id = '{domain['domain_id']}'
            LIMIT 1
        """
        try:
            existing = await self.database.execute(check_query)
            if not existing:
                # INSERT
                insert_query = f"""
                    INSERT INTO domains (
                        domain_id, domain_name, domain_code,
                        parent_domain, description, popularity,
                        source, discovery_time
                    )
                    VALUES (
                        '{domain['domain_id']}',
                        '{domain['domain_name'].replace("'", "''")}',
                        '{domain['domain_code']}',
                        '{domain.get('parent_domain', '')}',
                        '{domain.get('description', '').replace("'", "''")}',
                         {domain.get('popularity', 0)},
                        '{domain.get('source', 'default')}',
                        '{domain.get('discovery_time', time.strftime('%Y-%m-%d %H:%M:%S'))}'
                    )
                """
                await self.database.execute(insert_query)
                logger.debug(f"DomainDataAccess: حوزه جدید ذخیره شد: {domain['domain_id']}")
            else:
                # UPDATE
                update_query = f"""
                    ALTER TABLE domains UPDATE
                        domain_name = '{domain.get('domain_name', '').replace("'", "''")}',
                        domain_code = '{domain.get('domain_code', '')}',
                        parent_domain = '{domain.get('parent_domain', '')}',
                        description = '{domain.get('description', '').replace("'", "''")}',
                        popularity = {domain.get('popularity', 0)},
                        source = '{domain.get('source', 'default')}'
                    WHERE domain_id = '{domain['domain_id']}'
                """
                await self.database.execute(update_query)
                logger.debug(f"DomainDataAccess: حوزه موجود به‌روزرسانی شد: {domain['domain_id']}")

            # کش کردن در Redis به صورت یک کلید = JSON
            redis_key = f"domain:{domain['domain_id']}"
            await self.redis.set(redis_key, json.dumps(domain))
            return True

        except Exception as e:
            logger.error(f"DomainDataAccess: خطا در ذخیره حوزه: {e}")
            return False

    # متد کمکی برای جستجوی یک حوزه با استفاده از domain_code
    async def find_domain_by_code(self, domain_code: str) -> Optional[Dict[str, Any]]:
        """
        یافتن حوزه بر اساس domain_code
        """
        query = f"""
            SELECT *
            FROM domains
            WHERE domain_code='{domain_code}'
            LIMIT 1
        """
        try:
            rows = await self.database.execute(query)
            if rows:
                return rows[0]
            return None
        except Exception as e:
            logger.error(f"DomainDataAccess: خطا در find_domain_by_code({domain_code}): {e}")
            return None

    async def update_domain_popularity(self, domain_id: str, new_popularity: float) -> bool:
        """
        به‌روزرسانی فیلد popularity برای یک حوزه
        """
        query = f"""
            ALTER TABLE domains UPDATE
                popularity = {new_popularity}
            WHERE domain_id = '{domain_id}'
        """
        try:
            await self.database.execute(query)
            return True
        except Exception as e:
            logger.error(f"DomainDataAccess: خطا در update_domain_popularity({domain_id}): {e}")
            return False

    async def update_domain_parent(self, domain_id: str, new_parent_id: str) -> bool:
        """
        به‌روزرسانی فیلد parent_domain برای یک حوزه
        """
        query = f"""
            ALTER TABLE domains UPDATE
                parent_domain = '{new_parent_id}'
            WHERE domain_id = '{domain_id}'
        """
        try:
            await self.database.execute(query)
            return True
        except Exception as e:
            logger.error(f"DomainDataAccess: خطا در update_domain_parent({domain_id}): {e}")
            return False

    async def find_subdomains(self, parent_domain_id: str) -> List[Dict[str, Any]]:
        """
        یافتن زیرحوزه‌هایی که parent_domain آنها برابر با parent_domain_id است.
        """
        query = f"""
            SELECT *
            FROM domains
            WHERE parent_domain='{parent_domain_id}'
        """
        try:
            rows = await self.database.execute(query)
            return rows if rows else []
        except Exception as e:
            logger.error(f"DomainDataAccess: خطا در find_subdomains({parent_domain_id}): {e}")
            return []

    # ------------------------------------------------------------------------
    # --------------------------- CONCEPTS ------------------------------------
    # ------------------------------------------------------------------------

    async def load_concepts(self) -> Dict[str, Dict[str, Any]]:
        """
        بارگذاری مفاهیم تخصصی از پایگاه داده.
        """
        query = """
            SELECT
                concept_id,
                domain_id,
                concept_name,
                definition,
                examples,
                confidence,
                source,
                usage_count,
                discovery_time
            FROM domain_concepts
        """
        try:
            rows = await self.database.execute(query)
            if rows:
                concepts = {row['concept_id']: row for row in rows}
                logger.info(f"DomainDataAccess: {len(concepts)} مفهوم بازیابی شدند.")
                return concepts
            else:
                logger.info("DomainDataAccess: هیچ مفهومی یافت نشد.")
                return {}
        except Exception as e:
            logger.error(f"DomainDataAccess: خطا در بارگذاری مفاهیم: {e}")
            return {}

    async def store_concept(self, concept: Dict[str, Any]) -> bool:
        """
        ذخیره یا به‌روزرسانی یک مفهوم در پایگاه داده.
        """
        check_query = f"""
            SELECT concept_id FROM domain_concepts
            WHERE concept_id='{concept['concept_id']}'
            LIMIT 1
        """
        try:
            existing = await self.database.execute(check_query)
            if not existing:
                # INSERT
                examples_array_sql = await self.array_to_sql(concept.get('examples', []))
                insert_query = f"""
                    INSERT INTO domain_concepts (
                        concept_id, domain_id, concept_name,
                        definition, examples, confidence,
                        source, usage_count, discovery_time
                    )
                    VALUES (
                        '{concept['concept_id']}',
                        '{concept['domain_id']}',
                        '{concept['concept_name'].replace("'", "''")}',
                        '{concept.get('definition','').replace("'", "''")}',
                        {examples_array_sql},
                         {concept.get('confidence', 0)},
                        '{concept.get('source','default')}',
                         {concept.get('usage_count', 1)},
                        '{concept.get('discovery_time', time.strftime('%Y-%m-%d %H:%M:%S'))}'
                    )
                """
                await self.database.execute(insert_query)
                logger.debug(f"DomainDataAccess: مفهوم جدید ذخیره شد: {concept['concept_id']}")
            else:
                # UPDATE
                examples_array_sql = await self.array_to_sql(concept.get('examples', []))
                update_query = f"""
                    ALTER TABLE domain_concepts UPDATE
                        concept_name = '{concept.get('concept_name','').replace("'", "''")}',
                        definition = '{concept.get('definition','').replace("'", "''")}',
                        examples = {examples_array_sql},
                        confidence = {concept.get('confidence', 0)},
                        usage_count = {concept.get('usage_count', 1)},
                        source = '{concept.get('source','default')}'
                    WHERE concept_id = '{concept['concept_id']}'
                """
                await self.database.execute(update_query)
                logger.debug(f"DomainDataAccess: مفهوم موجود به‌روزرسانی شد: {concept['concept_id']}")

            # کش کردن در Redis
            redis_key = f"concept:{concept['concept_id']}"
            await self.redis.set(redis_key, json.dumps(concept))
            return True
        except Exception as e:
            logger.error(f"DomainDataAccess: خطا در ذخیره مفهوم: {e}")
            return False

    async def find_domain_concepts(self, domain_id: str) -> List[Dict[str, Any]]:
        """
        یافتن تمامی مفاهیم مربوط به یک domain_id
        """
        query = f"""
            SELECT *
            FROM domain_concepts
            WHERE domain_id='{domain_id}'
        """
        try:
            rows = await self.database.execute(query)
            return rows if rows else []
        except Exception as e:
            logger.error(f"DomainDataAccess: خطا در find_domain_concepts({domain_id}): {e}")
            return []

    async def update_concept_domain(self, concept_id: str, new_domain_id: str) -> bool:
        """
        تغییر domain_id یک مفهوم خاص
        """
        query = f"""
            ALTER TABLE domain_concepts UPDATE
                domain_id = '{new_domain_id}'
            WHERE concept_id = '{concept_id}'
        """
        try:
            await self.database.execute(query)
            return True
        except Exception as e:
            logger.error(f"DomainDataAccess: خطا در update_concept_domain({concept_id}): {e}")
            return False

    async def increment_concept_usage(self, concept_id: str) -> bool:
        """
        افزایش شمارنده استفاده از مفهوم (usage_count)
        """
        query = f"""
            ALTER TABLE domain_concepts UPDATE
                usage_count = usage_count + 1
            WHERE concept_id = '{concept_id}'
        """
        try:
            await self.database.execute(query)
            return True
        except Exception as e:
            logger.error(f"DomainDataAccess: خطا در increment_concept_usage({concept_id}): {e}")
            return False

    # ------------------------------------------------------------------------
    # --------------------------- RELATIONS -----------------------------------
    # ------------------------------------------------------------------------

    async def load_relations(self) -> Dict[str, Dict[str, Any]]:
        """
        بارگذاری تمامی روابط بین مفاهیم از پایگاه داده.
        """
        query = """
            SELECT
                relation_id,
                source_concept_id,
                target_concept_id,
                relation_type,
                description,
                confidence,
                source,
                usage_count,
                discovery_time
            FROM concept_relations
        """
        try:
            rows = await self.database.execute(query)
            if rows:
                relations = {row['relation_id']: row for row in rows}
                logger.info(f"DomainDataAccess: {len(relations)} رابطه بازیابی شدند.")
                return relations
            else:
                logger.info("DomainDataAccess: هیچ رابطه‌ای یافت نشد.")
                return {}
        except Exception as e:
            logger.error(f"DomainDataAccess: خطا در بارگذاری روابط: {e}")
            return {}

    async def store_relation(self, relation: Dict[str, Any]) -> bool:
        """
        ذخیره یا به‌روزرسانی یک رابطه بین مفاهیم در پایگاه داده.
        """
        check_query = f"""
            SELECT relation_id FROM concept_relations
            WHERE relation_id='{relation['relation_id']}'
            LIMIT 1
        """
        try:
            existing = await self.database.execute(check_query)
            if not existing:
                # INSERT
                insert_query = f"""
                    INSERT INTO concept_relations (
                        relation_id, source_concept_id, target_concept_id,
                        relation_type, description, confidence,
                        source, usage_count, discovery_time
                    )
                    VALUES (
                        '{relation['relation_id']}',
                        '{relation['source_concept_id']}',
                        '{relation['target_concept_id']}',
                        '{relation.get('relation_type','').replace("'", "''")}',
                        '{relation.get('description','').replace("'", "''")}',
                         {relation.get('confidence', 0)},
                        '{relation.get('source','default')}',
                         {relation.get('usage_count', 1)},
                        '{relation.get('discovery_time', time.strftime('%Y-%m-%d %H:%M:%S'))}'
                    )
                """
                await self.database.execute(insert_query)
                logger.debug(f"DomainDataAccess: رابطه جدید ذخیره شد: {relation['relation_id']}")
            else:
                # UPDATE
                update_query = f"""
                    ALTER TABLE concept_relations UPDATE
                        source_concept_id = '{relation.get('source_concept_id','')}',
                        target_concept_id = '{relation.get('target_concept_id','')}',
                        relation_type = '{relation.get('relation_type','').replace("'", "''")}',
                        description = '{relation.get('description','').replace("'", "''")}',
                        confidence = {relation.get('confidence', 0)},
                        usage_count = {relation.get('usage_count', 1)},
                        source = '{relation.get('source','default')}'
                    WHERE relation_id = '{relation['relation_id']}'
                """
                await self.database.execute(update_query)
                logger.debug(f"DomainDataAccess: رابطه موجود به‌روزرسانی شد: {relation['relation_id']}")

            # کش کردن در Redis
            redis_key = f"relation:{relation['relation_id']}"
            await self.redis.set(redis_key, json.dumps(relation))
            return True
        except Exception as e:
            logger.error(f"DomainDataAccess: خطا در ذخیره رابطه: {e}")
            return False

    async def find_relations_between(self, source_concept_id: str, target_concept_id: str, relation_type: str) -> List[Dict[str, Any]]:
        """
        یافتن روابط بین دو مفهوم مشخص با relation_type خاص
        """
        query = f"""
            SELECT *
            FROM concept_relations
            WHERE source_concept_id='{source_concept_id}'
              AND target_concept_id='{target_concept_id}'
              AND relation_type='{relation_type}'
        """
        try:
            rows = await self.database.execute(query)
            return rows if rows else []
        except Exception as e:
            logger.error(f"DomainDataAccess: خطا در find_relations_between(...): {e}")
            return []

    async def increment_relation_usage(self, relation_id: str) -> bool:
        """
        افزایش شمارنده استفاده از یک رابطه
        """
        query = f"""
            ALTER TABLE concept_relations UPDATE
                usage_count = usage_count + 1
            WHERE relation_id = '{relation_id}'
        """
        try:
            await self.database.execute(query)
            return True
        except Exception as e:
            logger.error(f"DomainDataAccess: خطا در increment_relation_usage({relation_id}): {e}")
            return False

    # ------------------------------------------------------------------------
    # --------------------------- ATTRIBUTES ----------------------------------
    # ------------------------------------------------------------------------

    async def load_attributes(self) -> Dict[str, Dict[str, Any]]:
        """
        بارگذاری تمامی ویژگی‌های مفاهیم از پایگاه داده.
        """
        query = """
            SELECT
                attribute_id,
                concept_id,
                attribute_name,
                attribute_value,
                description,
                confidence,
                source,
                usage_count,
                discovery_time
            FROM concept_attributes
        """
        try:
            rows = await self.database.execute(query)
            if rows:
                attributes = {row['attribute_id']: row for row in rows}
                logger.info(f"DomainDataAccess: {len(attributes)} ویژگی بازیابی شدند.")
                return attributes
            else:
                logger.info("DomainDataAccess: هیچ ویژگی‌ای یافت نشد.")
                return {}
        except Exception as e:
            logger.error(f"DomainDataAccess: خطا در بارگذاری ویژگی‌ها: {e}")
            return {}

    async def store_attribute(self, attribute: Dict[str, Any]) -> bool:
        """
        ذخیره یا به‌روزرسانی یک ویژگی برای یک مفهوم در پایگاه داده.
        """
        check_query = f"""
            SELECT attribute_id FROM concept_attributes
            WHERE attribute_id='{attribute['attribute_id']}'
            LIMIT 1
        """
        try:
            existing = await self.database.execute(check_query)
            if not existing:
                # INSERT
                insert_query = f"""
                    INSERT INTO concept_attributes (
                        attribute_id, concept_id, attribute_name,
                        attribute_value, description, confidence,
                        source, usage_count, discovery_time
                    )
                    VALUES (
                        '{attribute['attribute_id']}',
                        '{attribute['concept_id']}',
                        '{attribute.get('attribute_name','').replace("'", "''")}',
                        '{attribute.get('attribute_value','').replace("'", "''")}',
                        '{attribute.get('description','').replace("'", "''")}',
                         {attribute.get('confidence', 0)},
                        '{attribute.get('source','default')}',
                         {attribute.get('usage_count', 1)},
                        '{attribute.get('discovery_time', time.strftime('%Y-%m-%d %H:%M:%S'))}'
                    )
                """
                await self.database.execute(insert_query)
                logger.debug(f"DomainDataAccess: ویژگی جدید ذخیره شد: {attribute['attribute_id']}")
            else:
                # UPDATE
                update_query = f"""
                    ALTER TABLE concept_attributes UPDATE
                        concept_id = '{attribute.get('concept_id','')}',
                        attribute_name = '{attribute.get('attribute_name','').replace("'", "''")}',
                        attribute_value = '{attribute.get('attribute_value','').replace("'", "''")}',
                        description = '{attribute.get('description','').replace("'", "''")}',
                        confidence = {attribute.get('confidence', 0)},
                        usage_count = {attribute.get('usage_count', 1)},
                        source = '{attribute.get('source','default')}'
                    WHERE attribute_id = '{attribute['attribute_id']}'
                """
                await self.database.execute(update_query)
                logger.debug(f"DomainDataAccess: ویژگی موجود به‌روزرسانی شد: {attribute['attribute_id']}")

            # کش کردن در Redis
            redis_key = f"attribute:{attribute['attribute_id']}"
            await self.redis.set(redis_key, json.dumps(attribute))
            return True
        except Exception as e:
            logger.error(f"DomainDataAccess: خطا در ذخیره ویژگی: {e}")
            return False

    async def find_attributes_by_concept(self, concept_id: str) -> List[Dict[str, Any]]:
        """
        یافتن تمامی ویژگی‌های مربوط به یک مفهوم
        """
        query = f"""
            SELECT *
            FROM concept_attributes
            WHERE concept_id='{concept_id}'
        """
        try:
            rows = await self.database.execute(query)
            return rows if rows else []
        except Exception as e:
            logger.error(f"DomainDataAccess: خطا در find_attributes_by_concept({concept_id}): {e}")
            return []

    async def update_attribute_value(self, attribute_id: str, new_value: str) -> bool:
        """
        به‌روزرسانی مقدار یک ویژگی
        """
        query = f"""
            ALTER TABLE concept_attributes UPDATE
                attribute_value = '{new_value.replace("'", "''")}'
            WHERE attribute_id = '{attribute_id}'
        """
        try:
            await self.database.execute(query)
            return True
        except Exception as e:
            logger.error(f"DomainDataAccess: خطا در update_attribute_value({attribute_id}): {e}")
            return False


# --------------------------- نمونه تست مستقل -------------------------------
if __name__ == "__main__":
    import asyncio

    async def main():
        dda = DomainDataAccess()
        await dda.connect_resources()

        # مثال ساده از بارگذاری
        domains = await dda.load_domains()
        print("Domains:", json.dumps(domains, ensure_ascii=False, indent=2))

        # مثال ساده از ذخیره یک حوزه
        new_domain = {
            "domain_id": "d_test_01",
            "domain_name": "حوزه تستی",
            "domain_code": "TEST_DOMAIN",
            "parent_domain": "",
            "description": "این یک حوزه تستی است",
            "popularity": 0.5,
            "source": "script",
            "discovery_time": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        result = await dda.store_domain(new_domain)
        print("Store domain result:", result)

        # بارگذاری مجدد برای بررسی
        domains_after = await dda.load_domains()
        print("Domains (after storing):", json.dumps(domains_after, ensure_ascii=False, indent=2))

        await dda.disconnect_resources()

    asyncio.run(main())
